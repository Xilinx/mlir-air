#include "arm_bp_intf.h"

uint64_t get_bp_base_addr(uint8_t bp_id) {
  return (uint64_t)(BP_BASE_PADDR + bp_id * BP_SIZE);
}

// Templated Read/Write methods
// base_addr is the virtual address returned by an mmap call (e.g., from bp_map_cfg or bp_map_dram)
// offset is a byte offset into the mapped address range (e.g., BP_CFG_FREEZE_OFFSET)
// T is the data type to read/write (e.g., uint32_t or uint64_t)
// Each function first computes the proper virtual address and casts it to the appropriate type
// Then, the read or write is performed
template <class T>
void mmio_write(void* base_addr, off_t offset, T value) {
  T* addr = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base_addr) + offset);
  *addr = value;
}

template <class T>
T mmio_read(void* base_addr, off_t offset) {
  T* addr = reinterpret_cast<T*>(reinterpret_cast<uintptr_t>(base_addr) + offset);
  return *addr;
}

template <class T>
void mmio_write_all_bp_dram(off_t offset, T value) {
  for(int bp_id = 0; bp_id < NUM_HERD_CONTROLLERS; bp_id++) {
    void *bp_dram = (void *)(get_bp_base_addr(bp_id) + BP_DRAM_OFFSET);
    mmio_write<T>(bp_dram, offset, value);
  }
}

// byte reversal - works for integer data types
template <class T>
T reverse_bytes(T bytes) {
  int s = sizeof(T);
  T result = 0;
  // shift input right to select correct byte
  // shift selected byte left to place into result
  // start with LSB of bytes (placed into MSB of result)
  for (int i = 0; i < s; i++) {
    result |= (((bytes >> (i<<3)) & 0xff) << ((s-i-1)<<3));
  }
  return result;
}

void load_bp_dram(uint64_t phys_addr, uint32_t file_num_lines) {

  // Getting the physical address of where to write the firmware
  // expected file format:
  // @address
  // b0b1b2b3b4b5b6b7b7
  // b0b1b2b3b4b5b6b7b7
  // ...

  // ignore empty lines
  // expect no comments - each line is address or data
  // each data line is a power of two bytes, in hex

  if(phys_addr == NULL) return;

  off_t base_offset = BP_DRAM_OFFSET;
  off_t offset = 0;

  char *binary_str_array = (char *)phys_addr;

  char *token;
  token = strtok((char *)binary_str_array, "\n");

  for(int line_iter = 0; line_iter < file_num_lines; line_iter++) {

    // Creating a C++ string so can reuse BP host utilities
    std::string line(token);

    // Getting rid of the spaces in each line
    line.erase(std::remove(line.begin(), line.end(), ' '), line.end());

    // remove carriage returns at end of line
    while (line.back() == '\r') {
      line.pop_back();
    }
    // address line
    // parse the address and generate BP DRAM relative offset
    if (line.front() == '@') {
      line.erase(line.begin());
      offset = std::stoull(line, nullptr, 16) - base_offset;
    }
    // data line
    // write each byte in line to consecutive addresses, starting at (addr+offset)
    else {
      uint64_t val = 0;
      // hex characters on line
      int len = line.size();
      int i = 0;
      while (len > 0) {
        val = 0;
        int sublen = (len >= 16) ? 16 : len;
        // hex string of WWXXYYZZ...TT that needs to become TT...ZZYYXXWW
        // i.e., byte order reversal, but bits in each byte stay ordered
        std::string sub = line.substr(i,sublen);
        // convert hex string to 64-bit unsigned int
        val = stoull(std::string(sub.begin(), sub.end()), nullptr, 16);

        i += sublen;
        len -= sublen;

        uint64_t r = 0;
        if (sublen == 16) {
          r = reverse_bytes<uint64_t>(val);
          mmio_write_all_bp_dram<uint64_t>(offset, r);
          offset += 8;
        } else if (sublen == 8) {
          r = reverse_bytes<uint32_t>(val);
          mmio_write_all_bp_dram<uint32_t>(offset, r);
          offset += 4;
        } else if (sublen == 4) {
          r = reverse_bytes<uint16_t>(val);
          mmio_write_all_bp_dram<uint16_t>(offset, r);
          offset += 2;
        } else if (sublen == 2) {
          r = reverse_bytes<uint8_t>(val);
          mmio_write_all_bp_dram<uint8_t>(offset, r);
          offset += 1;
        }
      }
    }

    // Getting the next line if there is one
    if(line_iter == file_num_lines - 1) break;
    token = strtok(NULL, "\n");
  }

}

// Freezes every BP in the system
void bp_freeze() {
  for(int bp_iter = 0; bp_iter < NUM_HERD_CONTROLLERS; bp_iter++) {
    void *bp_cfg = (void *)(get_bp_base_addr(bp_iter) + BP_CFG_OFFSET);
    mmio_write<uint32_t>(bp_cfg, BP_CFG_FREEZE_OFFSET, BP_CFG_FREEZE_ON);
  }
}

// Strobes the reset to all of the BPs
void bp_strobe_reset() {
  void *bp_gpio = (void *)(BP_GPIO_PADDR + BP_GPIO_RESET_OFFSET);
  mmio_write<uint32_t>(bp_gpio, BP_GPIO_RESET_OFFSET, BP_GPIO_RESET_ON);
  sleep(1);
  mmio_write<uint32_t>(bp_gpio, BP_GPIO_RESET_OFFSET, BP_GPIO_RESET_OFF);
}


void handle_packet_prog_firmware(dispatch_packet_t *pkt) {
  packet_set_active(pkt, true);

  uint64_t  phys_addr       = pkt->arg[0];
  uint32_t  file_num_lines  = pkt->arg[1] & 0xFFFFFFFF;

  // Step 1: Put every BP in freeze
  bp_freeze();
  sleep(1);

  // Step 2: Strobe the reset
  bp_strobe_reset();
  sleep(1);

  // Step 3: Perform another reset
  bp_strobe_reset();

  // Step 4: Clear the DRAM of the BP
  for(int bp_iter = 0; bp_iter < NUM_HERD_CONTROLLERS; bp_iter++) {
    void *bp_dram = (void *)(get_bp_base_addr(bp_iter) + BP_DRAM_OFFSET);
    for(int offset = 0; offset < BP_DRAM_SIZE; offset+=8) {
      mmio_write<uint64_t>(bp_dram, offset, 0);
    }
  }

  // Step 5: Writing the contents to every BP in the system
  load_bp_dram(phys_addr, file_num_lines);

  // Step 6: Configure and unfreeze all of the BPs
  for(int bp_iter = 0; bp_iter < NUM_HERD_CONTROLLERS; bp_iter++) {
    void *bp_cfg = (void *)(get_bp_base_addr(bp_iter) + BP_CFG_OFFSET);
    mmio_write<uint32_t>(bp_cfg, BP_CFG_FREEZE_OFFSET, BP_CFG_FREEZE_OFF);

    // Sleeping as we wake them up so we see each BPs prints at the
    // the beginning.
    sleep(1);
  }
}

void start_bps() {
  // Go through every BP, set the proper configuration and unfreeze
  for(int bp_iter = 0; bp_iter < NUM_HERD_CONTROLLERS; bp_iter++) {
    void *bp_cfg = (void *)(get_bp_base_addr(bp_iter) + BP_CFG_OFFSET);
    mmio_write<uint32_t>(bp_cfg, BP_CFG_FREEZE_OFFSET, BP_CFG_FREEZE_OFF);

    // Sleeping as we wake them up so we see each BPs prints at the
    // the beginning.
    sleep(1);
  }
}
