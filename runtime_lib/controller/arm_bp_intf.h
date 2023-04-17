#ifndef __ARM_BP_INTF_H_
#define __ARM_BP_INTF_H_

#include "unistd.h"
#include <cstdint>
#include <cstring>
#include <string>
#include <algorithm>

#include "air_queue.h"
#include "hsa_defs.h"
#include "bp.h"
#include "xil_printf.h"

// Number of herd controllers in our current system
#define NUM_HERD_CONTROLLERS 6

// BP-specific platform addresses and size
#define BP_GPIO_PADDR             0x90000000000LL
#define BP_BASE_PADDR             0x80000000000LL
#define BP_GPIO_RESET_OFFSET      0x0LL
#define BP_GPIO_RESET_ON          0x0LL
#define BP_GPIO_RESET_OFF         0x1LL
#define BP_CFG_OFFSET             0x200000LL
#define BP_SIZE                   0x100000000LL

// Templated Read/Write methods
// base_addr is the virtual address returned by an mmap call (e.g., from bp_map_cfg or bp_map_dram)
// offset is a byte offset into the mapped address range (e.g., BP_CFG_FREEZE_OFFSET)
// T is the data type to read/write (e.g., uint32_t or uint64_t)
// Each function first computes the proper virtual address and casts it to the appropriate type
// Then, the read or write is performed
template <class T>
void mmio_write(void* base_addr, off_t offset, T value);

template <class T>
T mmio_read(void* base_addr, off_t offset);

template <class T>
void mmio_write_all_bp_dram(off_t offset, T value);

template <class T>
T reverse_bytes(T bytes);

uint64_t get_bp_base_addr(uint8_t bp_id);
void handle_packet_prog_firmware(dispatch_packet_t *pkt);
void start_bps();
void load_bp_dram(uint64_t phys_addr, uint32_t file_num_lines);
void bp_freeze();
void bp_strobe_reset();


#endif
