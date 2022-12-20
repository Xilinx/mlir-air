#ifndef AIRBIN_H
#define AIRBIN_H

#include <cstdint>
#include <fstream>

uint64_t airbin2mem(std::ifstream &infile, volatile uint32_t *tds_va,
                    uint32_t *tds_pa, volatile uint32_t *data_va,
                    uint32_t *data_pa, uint8_t col);

struct airbin_size {
  uint8_t start_col = 0;
  uint8_t num_cols = 1;
  uint8_t start_row = 1;
  uint8_t num_rows = 2;
};

airbin_size readairbinsize(std::ifstream &infile, uint8_t column_offset);
void readairbin(std::ifstream &infile);
#endif
