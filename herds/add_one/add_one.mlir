module {
  func @graph(%arg0: !torch.vtensor<[256], i32>) -> !torch.vtensor<[256], i32> {
    %0 = torch.constant.int 1
    %1 = "xten.add_constant"(%arg0, %0) : (!torch.vtensor<[256], i32>, !torch.int) -> !torch.vtensor<[256], i32>
    return %1 : !torch.vtensor<[256], i32>
  }
}