//===- PacketRoutingDomain.cpp ----------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#include "air/Util/PacketRoutingDomain.h"
#include "air/Util/Util.h"

#include "mlir/IR/Diagnostics.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

#define DEBUG_TYPE "air-packet-routing-domain"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::air;

namespace {

//===----------------------------------------------------------------------===//
// Static volume
//===----------------------------------------------------------------------===//

/// Volume of a single channel op's access pattern, in elements: the product of
/// its `sizes` when they are all static, else the whole memref. nullopt when
/// any size is dynamic -- an unknown volume must never be silently treated as
/// zero, since that would make a broadcast look like a partition.
static std::optional<int64_t> accessVolume(ChannelInterface chanOp) {
  SmallVector<OpFoldResult> sizes = chanOp.getMixedSizes();
  if (sizes.empty()) {
    auto memrefTy = dyn_cast<MemRefType>(chanOp.getMemref().getType());
    if (!memrefTy || !memrefTy.hasStaticShape())
      return std::nullopt;
    return (int64_t)getTensorVolume(memrefTy);
  }
  int64_t vol = 1;
  for (OpFoldResult s : sizes) {
    std::optional<int64_t> c = getConstantIntValue(s);
    if (!c)
      return std::nullopt;
    vol *= *c;
  }
  return vol;
}

/// Volume this op contributes across every execution of it, i.e. its access
/// volume scaled by the trip counts of the loops it sits in. nullopt if either
/// part is not static.
static std::optional<int64_t> totalVolume(ChannelInterface chanOp,
                                          Operation *scopeRoot) {
  std::optional<int64_t> vol = accessVolume(chanOp);
  if (!vol)
    return std::nullopt;
  std::optional<int64_t> trips =
      getStaticTripCountInRange(chanOp.getOperation(), scopeRoot);
  if (!trips)
    return std::nullopt;
  return *vol * *trips;
}

//===----------------------------------------------------------------------===//
// Channel facts
//===----------------------------------------------------------------------===//

/// A destination is the coordinate along the channel's BROADCAST dimension.
///
/// Not the whole index tuple: air-to-aie fans out along exactly one dimension
/// (specializeBroadcastShape keeps `getBroadcastDimension()` at its broadcast
/// extent and pins every other dimension to 1), so the remaining indices select
/// a bundle instance, not a destination. Keying on the full tuple would
/// multiply the destination count by the bundle size -- e.g. a `[NCX,1]` bundle
/// with `broadcast_shape=[NCX,NDEST]` would look like NCX*NDEST destinations
/// and misclassify the channel.
using DestKey = int64_t;

static std::optional<DestKey> destKeyOf(ChannelInterface chanOp,
                                        int broadcastDim) {
  OperandRange indices = chanOp.getIndices();
  if (broadcastDim < 0 || broadcastDim >= (int)indices.size())
    return std::nullopt;
  return getConstantIntValue(indices[broadcastDim]);
}

static bool isPacketChannel(ChannelOp chanOp) {
  auto ty = chanOp->getAttrOfType<StringAttr>("channel_type");
  return ty && ty.getValue() == "npu_dma_packet";
}

/// Classify one packet channel from its put/get volumes.
static PacketChannelFacts classify(ChannelOp chanOp,
                                   ArrayRef<ChannelInterface> puts,
                                   ArrayRef<ChannelInterface> gets,
                                   Operation *scopeRoot) {
  PacketChannelFacts c;

  // A channel's index tuple means one of two different things, and conflating
  // them turns every bundle into a spurious demux:
  //
  //   air.channel @toHub [4]                        -- BUNDLE: four parallel,
  //                                                    independent flows
  //   air.channel @outY [1,1] {broadcast_shape=[1,2]} -- one flow, two DESTS
  //
  // Only the broadcast dimension is a fan-out that a packet id can select
  // between. Without broadcast_shape each index is its own single-destination
  // flow, however many of them there are.
  if (!chanOp.isBroadcast()) {
    c.fanout = PacketFanout::SingleDest;
    c.numDests = 1;
    return c;
  }

  // Group the gets by destination (a coordinate along the broadcast dimension).
  int broadcastDim = chanOp.getBroadcastDimension();
  if (broadcastDim < 0) {
    c.reason = "broadcast channel with no resolvable broadcast dimension";
    return c;
  }
  llvm::MapVector<DestKey, int64_t> volByDest;
  for (ChannelInterface g : gets) {
    std::optional<DestKey> key = destKeyOf(g, broadcastDim);
    if (!key) {
      c.reason = "a get has a non-constant index on the broadcast dimension";
      return c;
    }
    std::optional<int64_t> v = totalVolume(g, scopeRoot);
    if (!v) {
      c.reason = "a get has a non-static volume";
      return c;
    }
    volByDest[*key] += *v;
  }
  c.numDests = volByDest.size();

  if (volByDest.empty()) {
    c.reason = "no gets";
    return c;
  }
  if (volByDest.size() == 1) {
    c.fanout = PacketFanout::SingleDest;
    return c;
  }

  int64_t putVol = 0;
  for (ChannelInterface p : puts) {
    std::optional<int64_t> v = totalVolume(p, scopeRoot);
    if (!v) {
      c.reason = "a put has a non-static volume";
      return c;
    }
    putVol += *v;
  }
  if (putVol == 0) {
    c.reason = "no puts";
    return c;
  }

  // Replication: every destination sees the whole stream.
  bool allFull =
      llvm::all_of(volByDest, [&](auto &kv) { return kv.second == putVol; });
  if (allFull) {
    c.fanout = PacketFanout::Broadcast;
    return c;
  }

  // Partition: the destinations' volumes sum to the stream.
  //
  // This is EVIDENCE, not proof. A partition is only physically routable if
  // something upstream writes a routing header per packet, and volumes cannot
  // say whether anything does. buildDomains() establishes that separately, from
  // a put naming a `dest`, and verify() rejects a partition nothing confirms.
  //
  // Requiring the header attribute HERE, as this used to, is what made the
  // attribute load-bearing for classification -- and therefore impossible for a
  // design to stop declaring.
  int64_t sum = 0;
  for (auto &kv : volByDest)
    sum += kv.second;

  // The producer side of a header-preserving hop carries a header word per
  // packet that the destination may strip, so the received total can fall
  // short of the sent total by that much. Accept a partition that accounts for
  // at least the payload and never exceeds what was sent.
  if (sum <= putVol && sum > 0) {
    c.fanout = PacketFanout::Demux;
    return c;
  }

  c.reason = "destination volumes neither replicate nor partition the put "
             "volume (sent " +
             std::to_string(putVol) + ", received " + std::to_string(sum) + ")";
  return c;
}
//===----------------------------------------------------------------------===//
// Forwarding-edge recovery
//===----------------------------------------------------------------------===//

/// Every block that encloses `op`, innermost first, mapped to the ancestor of
/// `op` that sits directly in it.
static void
collectAncestorChain(Operation *op,
                     llvm::SmallDenseMap<Block *, Operation *> &out) {
  for (Operation *o = op; o; o = o->getParentOp())
    if (Block *b = o->getBlock())
      out.try_emplace(b, o);
}

/// The innermost block enclosing both ops, with each one's ancestor in it.
/// Null when they share no block (different functions), or when both descend
/// from the SAME op in that block -- sibling regions of an scf.if are
/// alternatives with no order between them.
static Block *commonBlock(Operation *a, Operation *b, Operation *&aAnc,
                          Operation *&bAnc) {
  llvm::SmallDenseMap<Block *, Operation *> aChain;
  collectAncestorChain(a, aChain);
  for (Operation *o = b; o; o = o->getParentOp()) {
    auto it = aChain.find(o->getBlock());
    if (it == aChain.end())
      continue;
    if (it->second == o)
      return nullptr;
    aAnc = it->second;
    bAnc = o;
    return o->getBlock();
  }
  return nullptr;
}

/// Does `op` (or anything nested in it) overwrite the buffer `root` names?
///
/// air.channel.get is deliberately NOT a write for this purpose. A get deposits
/// bytes that arrived over a channel; it moves data, it does not transform it,
/// and whatever routing header those bytes carried they still carry. Several
/// gets gathering disjoint slices into one buffer before a single put sends it
/// onward is the ordinary shape of a memtile hop -- treating the second, third
/// and fourth gets as writes would sever exactly the chains this analysis
/// exists to follow.
///
/// Only writes whose target is KNOWN and resolves to `root` count. An opaque
/// write effect is ignored rather than assumed to hit `root`: guessing "yes"
/// here severs a chain, and a severed chain means a hop silently under-admits
/// ids, which on device is a hang. A false negative merely leaves today's
/// behavior in place.
static bool opWritesRoot(Operation *op, Value root) {
  bool found = false;
  op->walk([&](Operation *inner) {
    if (isa<ChannelGetOp>(inner))
      return WalkResult::skip();
    if (auto call = dyn_cast<CallOpInterface>(inner)) {
      // A call has no effect interface to consult, so any buffer it is handed
      // is assumed written. Kernels take their output buffer by reference.
      for (Value operand : call.getArgOperands()) {
        if (!isa<BaseMemRefType>(operand.getType()))
          continue;
        if (resolveBufferRoot(operand) == root) {
          found = true;
          return WalkResult::interrupt();
        }
      }
      return WalkResult::advance();
    }
    auto effectOp = dyn_cast<MemoryEffectOpInterface>(inner);
    if (!effectOp)
      return WalkResult::advance();
    SmallVector<MemoryEffects::EffectInstance> effects;
    effectOp.getEffects(effects);
    for (const auto &e : effects) {
      if (!isa<MemoryEffects::Write>(e.getEffect()))
        continue;
      Value target = e.getValue();
      if (target && resolveBufferRoot(target) == root) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

/// Is the payload `g` landed in `root` the same payload `p` sends onward?
///
/// Yes unless something between them rewrites the buffer. Note what is NOT
/// required: that `g` precede `p`. A rolled hop puts this iteration's buffer
/// before getting the next one, and demanding program order would sever it.
static bool isForwardingPair(ChannelInterface g, ChannelInterface p,
                             Value root) {
  Operation *gAnc = nullptr, *pAnc = nullptr;
  Block *blk = commonBlock(g.getOperation(), p.getOperation(), gAnc, pAnc);
  if (!blk)
    return true; // no order to reason about; keep the link
  Operation *first = gAnc, *last = pAnc;
  if (!first->isBeforeInBlock(last))
    std::swap(first, last);
  for (Operation *o = first->getNextNode(); o && o != last;
       o = o->getNextNode())
    if (opWritesRoot(o, root))
      return false;
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// PacketRoutingDomain
//===----------------------------------------------------------------------===//

namespace xilinx {
namespace air {

const PacketChannelFacts PacketRoutingDomainAnalysis::unknownFacts = {};

StringRef getPacketFanoutName(PacketFanout f) {
  switch (f) {
  case PacketFanout::Broadcast:
    return "broadcast";
  case PacketFanout::Demux:
    return "demux";
  case PacketFanout::SingleDest:
    return "single-destination";
  case PacketFanout::Unknown:
    return "unknown";
  }
  return "unknown";
}

bool PacketRoutingDomain::contains(ChannelOp c) const {
  if (demux == c)
    return true;
  return llvm::is_contained(hops, c);
}

PacketRoutingDomainAnalysis::PacketRoutingDomainAnalysis(ModuleOp mod)
    : mod(mod) {
  classifyChannels(mod);
  buildForwardingEdges(mod);
  buildDomains();
}

void PacketRoutingDomainAnalysis::classifyChannels(ModuleOp mod) {
  llvm::MapVector<StringRef, SmallVector<ChannelInterface>> putsOfByName,
      getsOfByName;
  mod.walk([&](ChannelInterface op) {
    if (isa<ChannelPutOp>(op.getOperation()))
      putsOfByName[op.getChanName()].push_back(op);
    else if (isa<ChannelGetOp>(op.getOperation()))
      getsOfByName[op.getChanName()].push_back(op);
  });
  mod.walk([&](ChannelOp chanOp) {
    if (!isPacketChannel(chanOp))
      return;
    packetChans.push_back(chanOp);
    StringRef name = chanOp.getSymName();
    putsOf[chanOp.getOperation()] = putsOfByName[name];
    getsOf[chanOp.getOperation()] = getsOfByName[name];
    facts[chanOp.getOperation()] = classify(
        chanOp, putsOfByName[name], getsOfByName[name], mod.getOperation());
  });
}

void PacketRoutingDomainAnalysis::buildForwardingEdges(ModuleOp mod) {
  // A hop is plain SSA: a get lands the payload in a buffer and a put on the
  // next channel sends that same buffer onward. Key on the RESOLVED root, not
  // the raw memref operand -- a subview, an air.execute result or a herd block
  // argument names the same bytes through a different Value, and keying on
  // identity drops those links without a word.
  llvm::DenseMap<Value, SmallVector<ChannelInterface>> puttersByRoot;
  for (ChannelOp c : packetChans)
    for (ChannelInterface p : putsOf[c.getOperation()])
      puttersByRoot[resolveBufferRoot(p.getMemref())].push_back(p);

  for (ChannelOp c : packetChans) {
    StringRef name = c.getSymName();
    llvm::SmallPtrSet<Operation *, 4> seen;
    for (ChannelInterface g : getsOf[c.getOperation()]) {
      Value root = resolveBufferRoot(g.getMemref());
      for (ChannelInterface p : puttersByRoot.lookup(root)) {
        if (p.getChanName() == name)
          continue;
        ChannelOp succ = getChannelDeclarationThroughSymbol(p);
        if (!succ || seen.contains(succ.getOperation()))
          continue;
        // Dedupe on the link that was ACCEPTED, not on the one first
        // considered. Marking `succ` seen before the forwarding test would let
        // one rejected pair veto a later pair that does forward -- the two
        // sides are related many-to-many, and only one of them has to hold.
        if (!isForwardingPair(g, p, root))
          continue;
        seen.insert(succ.getOperation());
        feeds[c.getOperation()].push_back(succ);
      }
    }
  }
}

void PacketRoutingDomainAnalysis::buildDomains() {
  // Reverse edges, so a demux can pull in everything that feeds it.
  llvm::DenseMap<Operation *, SmallVector<ChannelOp>> fedBy;
  for (ChannelOp c : packetChans)
    for (ChannelOp succ : feeds.lookup(c.getOperation()))
      fedBy[succ.getOperation()].push_back(c);

  auto pinnedIds = [](ChannelOp c) -> SmallVector<int64_t> {
    SmallVector<int64_t> ids;
    auto attr = c.getPacketIDs();
    if (!attr)
      return ids;
    for (Attribute a : attr)
      if (auto i = dyn_cast<IntegerAttr>(a))
        ids.push_back(i.getInt());
    return ids;
  };

  // Everything each dest-carrying put can reach downstream.
  //
  // A put naming a `dest` is choosing, at run time, which leaf this packet is
  // for. Nothing else it could mean -- so the packet channel it eventually
  // reaches covers its broadcast dimension over TIME, and that is the whole
  // space-vs-time question, answered by the design without an attribute for it.
  struct Origin {
    ChannelPutOp put;
    llvm::SmallPtrSet<Operation *, 8> reaches;
  };
  SmallVector<Origin> origins;
  {
    ModuleOp m = mod;
    m.walk([&](ChannelPutOp put) {
      if (!put.getDest())
        return;
      ChannelOp c0 = getChannelDeclarationThroughSymbol(put);
      if (!c0)
        return;
      Origin o;
      o.put = put;
      o.reaches.insert(c0.getOperation());
      SmallVector<ChannelOp> wl{c0};
      while (!wl.empty()) {
        ChannelOp cur = wl.pop_back_val();
        for (ChannelOp succ : feeds.lookup(cur.getOperation()))
          if (o.reaches.insert(succ.getOperation()).second)
            wl.push_back(succ);
      }
      origins.push_back(std::move(o));
    });
  }

  // One domain per demux. Ids belong to the domain, so this is also the only
  // place they are handed out.
  for (ChannelOp d : packetChans) {
    if (getFacts(d).fanout != PacketFanout::Demux)
      continue;

    PacketRoutingDomain dom;
    dom.demux = d;

    // Everything that can reach the demux, with no attribute gate.
    llvm::SmallPtrSet<Operation *, 8> upstream;
    {
      SmallVector<ChannelOp> wl{d};
      while (!wl.empty()) {
        ChannelOp cur = wl.pop_back_val();
        for (ChannelOp pred : fedBy.lookup(cur.getOperation()))
          if (upstream.insert(pred.getOperation()).second)
            wl.push_back(pred);
      }
    }

    // A hop lies on a path from an originating put to this demux: reachable
    // FORWARD from the put and BACKWARD from the demux. Intersecting the two
    // is what keeps an unrelated channel that merely happens to feed the same
    // buffer out of the domain -- the old attribute gate excluded those only
    // incidentally.
    llvm::SmallPtrSet<Operation *, 8> onPath;
    for (const Origin &o : origins) {
      if (!o.reaches.contains(d.getOperation()))
        continue;
      dom.originators.push_back(o.put);
      for (Operation *c : o.reaches)
        if (upstream.contains(c))
          onPath.insert(c);
    }
    // Collect by walking UPSTREAM from the demux, filtered to the path, so the
    // order means something. Iterating packetChans here would collect in
    // declaration order, which the std::reverse below then INVERTS -- in a
    // simple chain that yields exactly the reverse of the travel order the
    // field is documented to hold.
    {
      SmallVector<ChannelOp> wl{d};
      llvm::SmallPtrSet<Operation *, 8> visited{d.getOperation()};
      while (!wl.empty()) {
        ChannelOp cur = wl.pop_back_val();
        for (ChannelOp pred : fedBy.lookup(cur.getOperation())) {
          if (!onPath.contains(pred.getOperation()))
            continue;
          if (!visited.insert(pred.getOperation()).second)
            continue;
          dom.hops.push_back(pred);
          wl.push_back(pred);
        }
      }
    }

    if (dom.originators.empty()) {
      // No put names a dest anywhere upstream. Either the design predates
      // `dest` and declares its own header ownership, or the chain is broken.
      // Fall back to the attribute-gated walk so such designs keep working;
      // verify() is what tells the two apart.
      //
      // Deliberately NOT `continue` when the attribute is absent too: a
      // partition with no header writer anywhere is a broken design, and
      // forming the (empty) domain is what lets verify() say so instead of
      // silently declining to allocate and leaving air-to-aie to auto-assign a
      // single id that half the destinations will never match.
      SmallVector<ChannelOp> worklist{d};
      llvm::SmallPtrSet<Operation *, 8> visited{d.getOperation()};
      while (!worklist.empty()) {
        ChannelOp cur = worklist.pop_back_val();
        for (ChannelOp pred : fedBy.lookup(cur.getOperation())) {
          if (!channelSourceWritesHeader(pred))
            continue;
          if (!visited.insert(pred.getOperation()).second)
            continue;
          dom.hops.push_back(pred);
          worklist.push_back(pred);
        }
      }
    }
    // Furthest upstream first: the order the packet actually travels, which is
    // the order a reader of the report wants.
    std::reverse(dom.hops.begin(), dom.hops.end());

    SmallVector<int64_t> declared = pinnedIds(d);
    if (!declared.empty()) {
      dom.ids = declared;
      dom.idsDeclared = true;
    } else {
      unsigned n = getFacts(d).numDests;
      if (n >= 2 && n <= (unsigned)kMaxPacketID + 1)
        // From the TOP of the id space. air-to-aie hands out the lowest free id
        // and treats a pinned one as claimed, so allocating upward would
        // renumber every other packet channel and perturb a tuned floorplan.
        for (unsigned k = 0; k < n; ++k)
          dom.ids.push_back(kMaxPacketID - k);
    }

    unsigned idx = domains.size();
    for (ChannelOp m : dom.hops) {
      auto it = domainIdxOf.find(m.getOperation());
      if (it != domainIdxOf.end()) {
        faults.push_back(
            {m.getOperation(),
             ("packet channel @" + m.getSymName() +
              " forwards into two "
              "routing domains, so there is no single set of ids its switchbox "
              "could admit")
                 .str(),
             {("one is the demux @" + domains[it->second].demux.getSymName())
                  .str(),
              ("the other is the demux @" + d.getSymName()).str()}});
        continue;
      }
      domainIdxOf[m.getOperation()] = idx;
    }
    domainIdxOf[d.getOperation()] = idx;
    domains.push_back(std::move(dom));
  }
}

const PacketRoutingDomain *
PacketRoutingDomainAnalysis::getDomainOf(ChannelOp c) const {
  auto it = domainIdxOf.find(c.getOperation());
  return it == domainIdxOf.end() ? nullptr : &domains[it->second];
}

unsigned PacketRoutingDomainAnalysis::getInferredIdCount(ChannelOp c) const {
  const PacketChannelFacts &f = getFacts(c);
  // A demux's own count comes from its destinations. Reading it off the domain
  // would read back a declared list and make every pin agree with itself.
  if (f.fanout == PacketFanout::Demux)
    return f.numDests;
  if (const PacketRoutingDomain *dom = getDomainOf(c))
    if (!dom->ids.empty())
      return dom->ids.size();
  return f.fanout == PacketFanout::Unknown ? 0 : 1;
}

const PacketChannelFacts &
PacketRoutingDomainAnalysis::getFacts(ChannelOp c) const {
  auto it = facts.find(c.getOperation());
  return it == facts.end() ? unknownFacts : it->second;
}

ArrayRef<ChannelInterface>
PacketRoutingDomainAnalysis::getPuts(ChannelOp c) const {
  auto it = putsOf.find(c.getOperation());
  return it == putsOf.end() ? ArrayRef<ChannelInterface>() : it->second;
}

ArrayRef<ChannelInterface>
PacketRoutingDomainAnalysis::getGets(ChannelOp c) const {
  auto it = getsOf.find(c.getOperation());
  return it == getsOf.end() ? ArrayRef<ChannelInterface>() : it->second;
}

//===----------------------------------------------------------------------===//
// verify()
//===----------------------------------------------------------------------===//

/// Is this put issued by something that could have CHOSEN the destination?
///
/// Only a compute core can. A put reading an L2/L3 buffer is a pure data
/// mover: it forwards bytes it did not produce and cannot know where they are
/// meant to go -- which is the whole reason the decision travels in the packet
/// header instead. So a demux fed from L2 with nothing upstream of it has lost
/// its chain, however well-formed it looks locally.
static bool putCouldHaveWrittenHeader(ChannelInterface put) {
  auto ty = dyn_cast<BaseMemRefType>(put.getMemref().getType());
  return ty && isL1(ty);
}

LogicalResult PacketRoutingDomainAnalysis::verify() const {
  bool failed = false;
  ModuleOp m = mod; // walk() is non-const

  auto emit = [&](Operation *at, const Twine &msg,
                  ArrayRef<std::string> notes = {}) {
    InFlightDiagnostic diag = at->emitOpError(msg);
    for (const std::string &n : notes)
      diag.attachNote() << n;
    failed = true;
  };

  for (const Fault &f : faults)
    emit(f.at, f.message, f.notes);

  // A put that names a destination but has no demux to select between.
  //
  // The old diagnostic for this blamed the put's own channel for "not being a
  // demux" -- but a hop is not supposed to be one, and the real fault is
  // usually a severed link further downstream. Name the domain instead.
  m.walk([&](ChannelPutOp put) {
    if (!put.getDest())
      return;
    ChannelOp c = getChannelDeclarationThroughSymbol(put);
    if (!c) {
      emit(put, "names a channel that does not resolve");
      return;
    }
    const PacketRoutingDomain *dom = getDomainOf(c);
    if (dom && dom->ids.size() >= 2)
      return;

    SmallVector<std::string> notes;
    if (!dom) {
      notes.push_back(
          ("@" + c.getSymName() +
           " reaches no packet channel with more than one destination, so "
           "nothing downstream would route on the header this put writes")
              .str());
      SmallVector<ChannelOp> succs = feeds.lookup(c.getOperation());
      if (succs.empty()) {
        notes.push_back(("no packet channel forwards @" + c.getSymName() +
                         "'s payload onward; the chain is recovered by "
                         "matching the buffer a get lands in against the "
                         "buffer a later put sends")
                            .str());
      } else {
        std::string chain;
        for (ChannelOp s : succs)
          chain += (chain.empty() ? "" : ", ") + ("@" + s.getSymName()).str();
        notes.push_back("it forwards into " + chain +
                        ", none of which is a demux either");
      }
    } else {
      ChannelOp d = dom->demux;
      notes.push_back(
          ("the demux @" + d.getSymName() + " has " +
           std::to_string(dom->ids.size()) +
           " routing id(s); selecting a destination needs at least 2")
              .str());
    }
    emit(put, "selects a destination, but its routing domain has no demux",
         notes);
  });

  // A demux fed from L2 with no upstream at all.
  //
  // This is the failure the whole analysis exists for, and it is the one that
  // used to be completely silent: break the chain feeding a demux and the hops
  // simply never learn its ids, their switchboxes filter the extra ids out
  // mid-route, the demux never fires and the device times out. Nothing in the
  // IR looks wrong locally.
  for (const PacketRoutingDomain &domRef : domains) {
    PacketRoutingDomain dom = domRef;
    if (!dom.originators.empty() || !dom.hops.empty())
      continue;
    // A declared id list is the design asserting it knows its own routing --
    // possibly via a header source this analysis cannot see. The fault below
    // is about a list the compiler DERIVED, where a wrong derivation is ours.
    if (dom.idsDeclared)
      continue;
    ArrayRef<ChannelInterface> puts = getPuts(dom.demux);
    if (puts.empty() || llvm::any_of(puts, putCouldHaveWrittenHeader))
      continue;
    emit(dom.demux.getOperation(),
         "is a packet demux whose routing header nothing upstream writes",
         {"its put reads an L2/L3 buffer, so it forwards bytes it did not "
          "produce and cannot have chosen a destination",
          "no put naming a dest, and no forwarding hop, reaches it -- the "
          "chain is recovered by matching the buffer a get lands in against "
          "the buffer a later put sends, so a subview, a copy or a compute "
          "between the two severs it"});
  }

  // A pinned list that disagrees with its domain. An explicit declaration is an
  // assertion to check, never a second source of truth.
  auto renderIds = [](ArrayRef<int64_t> ids) {
    std::string out;
    for (int64_t id : ids)
      out += (out.empty() ? "" : ", ") + std::to_string(id);
    return "[" + out + "]";
  };
  for (const PacketRoutingDomain &domRef : domains) {
    PacketRoutingDomain dom = domRef;
    SmallVector<int64_t> sortedWant(dom.ids);
    llvm::sort(sortedWant);
    for (ChannelOp hop : dom.hops) {
      auto attr = hop.getPacketIDs();
      if (!attr)
        continue;
      // Compare the id VALUES, not just how many there are. A hop pinning
      // {1,2} for a demux routing {3,4} has the right cardinality and entirely
      // the wrong routes: its switchbox admits two ids, neither of which the
      // demux keys on, so every packet is filtered out mid-route. Counting
      // alone would wave that through -- and a pin agreeing with itself is the
      // precise shape of "two spellings, no link" this analysis exists to
      // remove.
      //
      // Order is NOT compared. A hop is single-destination, so air-to-aie
      // returns its whole list for the one buffer and the sequence carries no
      // meaning. On a demux order does matter (destination i routes with
      // ids[i]), but a demux's list IS the domain's by construction, so there
      // is nothing to disagree with.
      SmallVector<int64_t> pinned;
      for (Attribute a : attr)
        if (auto i = dyn_cast<IntegerAttr>(a))
          pinned.push_back(i.getInt());
      SmallVector<int64_t> sortedPinned(pinned);
      llvm::sort(sortedPinned);
      if (sortedPinned == sortedWant)
        continue;
      emit(hop.getOperation(),
           "pins routing ids " + renderIds(pinned) +
               ", but forwards for a demux routing " + renderIds(dom.ids),
           {("the demux is @" + dom.demux.getSymName()).str(),
            "a hop must admit exactly the ids the demux keys on -- any id it "
            "omits is filtered out at its own switchbox and never reaches the "
            "demux",
            "the order of a hop's list is not checked: it is "
            "single-destination, so air-to-aie returns the whole list for its "
            "one buffer"});
    }
  }

  return mlir::failure(failed);
}

//===----------------------------------------------------------------------===//
// Reporting
//===----------------------------------------------------------------------===//

void PacketRoutingDomainAnalysis::printReport(llvm::raw_ostream &os) const {
  for (auto [i, domRef] : llvm::enumerate(domains)) {
    PacketRoutingDomain dom = domRef;
    os << "packet routing domain #" << i << ": " << dom.ids.size()
       << " id(s) [";
    llvm::interleaveComma(dom.ids, os);
    os << "] (" << (dom.idsDeclared ? "declared" : "allocated") << ")\n";
    for (ChannelPutOp p : dom.originators)
      os << "  originator  " << p.getChanName() << " (dest)\n";
    for (ChannelOp h : dom.hops)
      os << "  hop         @" << h.getSymName() << " (" << getFacts(h).numDests
         << " dest)\n";
    os << "  demux       @" << dom.demux.getSymName() << " ("
       << getFacts(dom.demux).numDests << " dests)\n";
  }
}

void PacketRoutingDomainAnalysis::emitReportRemarks() const {
  for (ChannelOp c : packetChans) {
    const PacketChannelFacts &f = getFacts(c);
    InFlightDiagnostic note = c->emitRemark();
    note << "packet channel @" << c.getSymName() << ": "
         << getPacketFanoutName(f.fanout) << " over " << f.numDests
         << " destination(s)";
    // A channel outside any demux domain still needs one id of its own --
    // unless it could not be classified at all, where the honest answer is to
    // claim nothing.
    const PacketRoutingDomain *dom = getDomainOf(c);
    unsigned inferred = getInferredIdCount(c);
    if (inferred) {
      note << "; infers " << inferred << " routing id(s)";
      if (dom && !dom->ids.empty()) {
        ChannelOp d = dom->demux;
        if (d != c)
          note << " (forwarded from the demux @" << d.getSymName() << ")";
      }
    }
    if (auto pinned = c.getPacketIDs())
      note << "; pins " << pinned.size();
    // Spell the chain out on the demux. The hop ORDER is a documented property
    // of the domain and was silently wrong once; printing it is what lets a
    // test pin it.
    if (dom && dom->demux == c && !dom->hops.empty()) {
      note << "; fed by ";
      bool first = true;
      for (ChannelOp h : dom->hops) {
        note << (first ? "@" : " -> @") << h.getSymName();
        first = false;
      }
    }
    if (!f.reason.empty())
      note << " [" << f.reason << "]";
  }
}

} // namespace air
} // namespace xilinx
