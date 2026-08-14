//===- PacketRoutingDomain.h ------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

#ifndef AIR_UTIL_PACKET_ROUTING_DOMAIN_H
#define AIR_UTIL_PACKET_ROUTING_DOMAIN_H

#include "air/Dialect/AIR/AIRDialect.h"

#include "mlir/IR/BuiltinOps.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx {
namespace air {

//===----------------------------------------------------------------------===//
// Per-channel facts
//===----------------------------------------------------------------------===//

/// How one packet channel fans out.
enum class PacketFanout {
  Broadcast,  // every destination receives every packet -> 1 id
  Demux,      // the destinations partition the stream -> one id per destination
  SingleDest, // nothing to decide
  Unknown,    // not statically classifiable; say so, never guess
};

StringRef getPacketFanoutName(PacketFanout f);

struct PacketChannelFacts {
  PacketFanout fanout = PacketFanout::Unknown;
  unsigned numDests = 0;
  /// Why the classification came out Unknown. Empty otherwise.
  std::string reason;
};

//===----------------------------------------------------------------------===//
// Routing domains
//===----------------------------------------------------------------------===//

/// The channels that carry ONE routing header end to end.
///
/// A packet's header is written once, at the point the destination is chosen,
/// and is then read by exactly one switchbox -- the demux -- somewhere
/// downstream. Every channel the packet crosses in between forwards that same
/// header untouched, and each of their switchboxes has to admit every id the
/// demux will key on. Miss one and the extra ids are filtered out mid-route,
/// the demux never fires, and the device hangs.
///
/// So ids are a property of the DOMAIN, not of a channel: allocated once, and
/// shared by the demux and every hop that feeds it.
struct PacketRoutingDomain {
  /// Puts carrying a `dest` operand -- the sites that pick a destination at run
  /// time and stamp the header. May be empty for a domain whose ids the front
  /// end pins and whose header some kernel writes by hand.
  SmallVector<ChannelPutOp> originators;
  /// Header-preserving single-destination channels that forward the packet,
  /// ordered from furthest upstream to nearest the demux.
  SmallVector<ChannelOp> hops;
  /// The channel whose switchbox actually routes on the header. Never null in a
  /// domain that verify() accepts.
  ChannelOp demux;
  /// The routing ids, allocated once for the whole domain.
  SmallVector<int64_t> ids;
  /// True when `ids` came from a front-end `packet_ids` pin rather than being
  /// allocated here. A pinned list is ORDERED (destination i routes with
  /// ids[i]) and that order is design input; an allocated one is not.
  bool idsDeclared = false;

  bool contains(ChannelOp c) const;
};

/// Which packet channels share a routing header, and what ids that header can
/// take.
///
/// Placement is load-bearing. The domain relation is recovered from plain SSA
/// -- a get lands a payload in a buffer and a put sends that same buffer onward
/// -- which only holds while the IR still looks like the front end wrote it.
/// Run this before `air-dma-to-channel` or `split-l2-memref` and the buffers
/// have been rewritten out from under the relation.
class PacketRoutingDomainAnalysis {
public:
  explicit PacketRoutingDomainAnalysis(ModuleOp mod);

  ArrayRef<PacketRoutingDomain> getDomains() const { return domains; }
  ArrayRef<ChannelOp> getPacketChannels() const { return packetChans; }

  /// The domain `c` belongs to, or null if it is in none.
  const PacketRoutingDomain *getDomainOf(ChannelOp c) const;
  const PacketChannelFacts &getFacts(ChannelOp c) const;

  /// How many routing ids `c`'s switchbox has to admit, derived independently
  /// of anything the channel declares. This is what a `packet_ids` pin is
  /// checked AGAINST, so it must never be read back off that pin.
  unsigned getInferredIdCount(ChannelOp c) const;

  ArrayRef<ChannelInterface> getPuts(ChannelOp c) const;
  ArrayRef<ChannelInterface> getGets(ChannelOp c) const;

  /// Emit a diagnostic for every structural fault found, and fail if any was.
  /// This is the difference between a misrouted packet showing up as a compile
  /// error and showing up as a device timeout.
  LogicalResult verify() const;

  /// One block per domain: originators, hops in order, the demux, the ids.
  void printReport(llvm::raw_ostream &os) const;
  /// Same, as op remarks, for `-air-annotate-packet-ids=report=true`.
  void emitReportRemarks() const;

private:
  void classifyChannels(ModuleOp mod);
  void buildForwardingEdges(ModuleOp mod);
  void buildDomains();

  ModuleOp mod;
  SmallVector<ChannelOp> packetChans;
  llvm::DenseMap<Operation *, PacketChannelFacts> facts;
  llvm::DenseMap<Operation *, SmallVector<ChannelInterface>> putsOf, getsOf;

  /// Channel -> the channels it forwards INTO (its successors downstream).
  llvm::DenseMap<Operation *, SmallVector<ChannelOp>> feeds;

  SmallVector<PacketRoutingDomain> domains;
  llvm::DenseMap<Operation *, unsigned> domainIdxOf;

  /// Faults found while building, replayed by verify(). Collected rather than
  /// emitted eagerly so that constructing the analysis stays side-effect free
  /// and a caller can choose to only report.
  struct Fault {
    Operation *at;
    std::string message;
    /// Extra lines attached to the diagnostic as notes.
    SmallVector<std::string> notes;
  };
  SmallVector<Fault> faults;

  static const PacketChannelFacts unknownFacts;
};

} // namespace air
} // namespace xilinx

#endif // AIR_UTIL_PACKET_ROUTING_DOMAIN_H
