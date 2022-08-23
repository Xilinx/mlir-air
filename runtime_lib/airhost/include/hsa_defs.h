// (c) Copyright 2022 Xilinx Inc. All Rights Reserved.

#ifndef _HSA_DEFS_H_
#define _HSA_DEFS_H_

/**
 * @brief Queue features.
 */
typedef enum {
  /**
   * Queue supports kernel dispatch packets.
   */
  HSA_QUEUE_FEATURE_KERNEL_DISPATCH = 1,

  /**
   * Queue supports agent dispatch packets.
   */
  HSA_QUEUE_FEATURE_AGENT_DISPATCH = 2
} hsa_queue_feature_t;


typedef enum {
  /**
   * Queue supports multiple producers. Use of multiproducer queue mechanics is
   * required.
   */
  HSA_QUEUE_TYPE_MULTI = 0,
  /**
   * Queue only supports a single producer. In some scenarios, the application
   * may want to limit the submission of AQL packets to a single agent. Queues
   * that support a single producer may be more efficient than queues supporting
   * multiple producers. Use of multiproducer queue mechanics is not supported.
   */
  HSA_QUEUE_TYPE_SINGLE = 1,
  /**
   * Queue supports multiple producers and cooperative dispatches. Cooperative
   * dispatches are able to use GWS synchronization. Queues of this type may be
   * limited in number. The runtime may return the same queue to serve multiple
   * ::hsa_queue_create calls when this type is given. Callers must inspect the
   * returned queue to discover queue size. Queues of this type are reference
   * counted and require a matching number of ::hsa_queue_destroy calls to
   * release. Use of multiproducer queue mechanics is required. See
   * ::HSA_AMD_AGENT_INFO_COOPERATIVE_QUEUES to query agent support for this
   * type.
   */
  HSA_QUEUE_TYPE_COOPERATIVE = 2
} hsa_queue_type_t;

/**
 * @brief Packet type.
 */
typedef enum {
  /**
   * Vendor-specific packet.
   */
  HSA_PACKET_TYPE_VENDOR_SPECIFIC = 0,
  /**
   * The packet has been processed in the past, but has not been reassigned to
   * the packet processor. A packet processor must not process a packet of this
   * type. All queues support this packet type.
   */
  HSA_PACKET_TYPE_INVALID = 1,
  /**
   * Packet used by agents for dispatching jobs to kernel agents. Not all
   * queues support packets of this type (see ::hsa_queue_feature_t).
   */
  HSA_PACKET_TYPE_KERNEL_DISPATCH = 2,
  /**
   * Packet used by agents to delay processing of subsequent packets, and to
   * express complex dependencies between multiple packets. All queues support
   * this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_AND = 3,
  /**
   * Packet used by agents for dispatching jobs to agents.  Not all
   * queues support packets of this type (see ::hsa_queue_feature_t).
   */
  HSA_PACKET_TYPE_AGENT_DISPATCH = 4,
  /**
   * Packet used by agents to delay processing of subsequent packets, and to
   * express complex dependencies between multiple packets. All queues support
   * this packet type.
   */
  HSA_PACKET_TYPE_BARRIER_OR = 5
} hsa_packet_type_t;

/**
 * @brief Sub-fields of the @a header field that is present in any AQL
 * packet. The offset (with respect to the address of @a header) of a sub-field
 * is identical to its enumeration constant. The width of each sub-field is
 * determined by the corresponding value in ::hsa_packet_header_width_t. The
 * offset and the width are expressed in bits.
 */
typedef enum {
  /**
   * Packet type. The value of this sub-field must be one of
   * ::hsa_packet_type_t. If the type is ::HSA_PACKET_TYPE_VENDOR_SPECIFIC, the
   * packet layout is vendor-specific.
   */
  HSA_PACKET_HEADER_TYPE = 0,
  /**
   * Barrier bit. If the barrier bit is set, the processing of the current
   * packet only launches when all preceding packets (within the same queue) are
   * complete.
   */
  HSA_PACKET_HEADER_BARRIER = 8,
  /**
   * Acquire fence scope. The value of this sub-field determines the scope and
   * type of the memory fence operation applied before the packet enters the
   * active phase. An acquire fence ensures that any subsequent global segment
   * or image loads by any unit of execution that belongs to a dispatch that has
   * not yet entered the active phase on any queue of the same kernel agent,
   * sees any data previously released at the scopes specified by the acquire
   * fence. The value of this sub-field must be one of ::hsa_fence_scope_t.
   */
  HSA_PACKET_HEADER_ACQUIRE_FENCE_SCOPE = 9,
  /**
   * Release fence scope, The value of this sub-field determines the scope and
   * type of the memory fence operation applied after kernel completion but
   * before the packet is completed. A release fence makes any global segment or
   * image data that was stored by any unit of execution that belonged to a
   * dispatch that has completed the active phase on any queue of the same
   * kernel agent visible in all the scopes specified by the release fence. The
   * value of this sub-field must be one of ::hsa_fence_scope_t.
   */
  HSA_PACKET_HEADER_RELEASE_FENCE_SCOPE = 11
} hsa_packet_header_t;

/**
 * @brief Width (in bits) of the sub-fields in ::hsa_packet_header_t.
 */
typedef enum {
  HSA_PACKET_HEADER_WIDTH_TYPE = 8,
  HSA_PACKET_HEADER_WIDTH_BARRIER = 1,
  HSA_PACKET_HEADER_WIDTH_ACQUIRE_FENCE_SCOPE = 2,
  HSA_PACKET_HEADER_WIDTH_RELEASE_FENCE_SCOPE = 2
} hsa_packet_header_width_t;

typedef enum {
  HSA_STATUS_SUCCESS = 0x0,

HSA_STATUS_INFO_BREAK = 0x1,

HSA_STATUS_ERROR = 0x1000,

HSA_STATUS_ERROR_INVALID_ARGUMENT = 0x1001,

HSA_STATUS_ERROR_INVALID_QUEUE_CREATION = 0x1002,

HSA_STATUS_ERROR_INVALID_ALLOCATION = 0x1003,

HSA_STATUS_ERROR_INVALID_AGENT = 0x1004,

HSA_STATUS_ERROR_INVALID_REGION = 0x1005,

HSA_STATUS_ERROR_INVALID_SIGNAL = 0x1006,

HSA_STATUS_ERROR_INVALID_QUEUE = 0x1007,

HSA_STATUS_ERROR_OUT_OF_RESOURCES = 0x1008,

HSA_STATUS_ERROR_INVALID_PACKET_FORMAT = 0x1009,

HSA_STATUS_ERROR_RESOURCE_FREE = 0x100A,

HSA_STATUS_ERROR_NOT_INITIALIZED = 0x100B,

HSA_STATUS_ERROR_REFCOUNT_OVERFLOW = 0x100C,

HSA_STATUS_ERROR_INCOMPATIBLE_ARGUMENTS = 0x100D,

HSA_STATUS_ERROR_INVALID_INDEX = 0x100E,

HSA_STATUS_ERROR_INVALID_ISA = 0x100F,

HSA_STATUS_ERROR_INVALID_ISA_NAME = 0x1017,

HSA_STATUS_ERROR_INVALID_CODE_OBJECT = 0x1010,

HSA_STATUS_ERROR_INVALID_EXECUTABLE = 0x1011,

HSA_STATUS_ERROR_FROZEN_EXECUTABLE = 0x1012,

HSA_STATUS_ERROR_INVALID_SYMBOL_NAME = 0x1013,

HSA_STATUS_ERROR_VARIABLE_ALREADY_DEFINED = 0x1014,

HSA_STATUS_ERROR_VARIABLE_UNDEFINED = 0x1015,

HSA_STATUS_ERROR_EXCEPTION = 0x1016,

HSA_STATUS_ERROR_INVALID_CODE_SYMBOL = 0x1018,

HSA_STATUS_ERROR_INVALID_EXECUTABLE_SYMBOL = 0x1019,

HSA_STATUS_ERROR_INVALID_FILE = 0x1020,

HSA_STATUS_ERROR_INVALID_CODE_OBJECT_READER = 0x1021,

HSA_STATUS_ERROR_INVALID_CACHE = 0x1022,

HSA_STATUS_ERROR_INVALID_WAVEFRONT = 0x1023,

HSA_STATUS_ERROR_INVALID_SIGNAL_GROUP = 0x1024,

HSA_STATUS_ERROR_INVALID_RUNTIME_STATE = 0x1025
} hsa_status_t;

typedef enum {
  HSA_SIGNAL_CONDITION_EQ = 0,
  HSA_SIGNAL_CONDITION_NE = 1,
  HSA_SIGNAL_CONDITION_LT = 2,
  HSA_SIGNAL_CONDITION_GTE = 3
} hsa_signal_condition_t;

typedef enum {
  HSA_WAIT_STATE_BLOCKED = 0,
  HSA_WAIT_STATE_ACTIVE = 1
} hsa_wait_state_t;

#endif
