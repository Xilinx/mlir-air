module {

func @foo(%arg0: i32) {
  %cst1 = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %cst1, %size_y = %cst1) attributes {link_with="beefmaker_kernel.o"} {
    %src0 = alloc() {sym_name="beef"}: memref<1024xi32, 2>
    call @beefmaker_kernel(%src0) : (memref<1024xi32, 2>) -> ()
    air.herd_terminator
  }
  return
} 

func private @beefmaker_kernel(%A: memref<1024xi32, 2>) -> ()

}
