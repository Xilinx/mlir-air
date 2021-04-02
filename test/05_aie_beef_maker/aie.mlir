
// A very minimal architecture ...

// aie-opt --aie-create-flows --aie-find-flows %s | aie-translate --aie-generate-xaie 

module {
  %t72 = AIE.tile(7, 2)

  AIE.core(%t72) {
    AIE.end
  }
  {
    elf_file = "maker.elf"
  }
  
  %buf72_0 = AIE.buffer(%t72) {sym_name="buffer"}: memref<4xi32>
  %l72_0 = AIE.lock(%t72, 0)
}
