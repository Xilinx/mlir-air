#include <boost/program_options.hpp>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

#define BATCH 2
#define CHOUT 64
#define CHIN 32
#define K 3
#define XIN 17
#define YIN 17
#define XOUT ((XIN - K) / 2 + 1)
#define YOUT ((YIN - K) / 2 + 1)

#define A_VOLUME (BATCH * CHIN * XIN * YIN)
#define B_VOLUME (CHIN * CHOUT * K * K)
#define C_VOLUME (BATCH * CHOUT * XOUT * YOUT)

#define A_DATATYPE int32_t
#define B_DATATYPE int32_t
#define C_DATATYPE int32_t

constexpr int A_SIZE = (A_VOLUME * sizeof(A_DATATYPE));
constexpr int B_SIZE = (B_VOLUME * sizeof(B_DATATYPE));
constexpr int C_SIZE = (C_VOLUME * sizeof(C_DATATYPE));
constexpr int TRACE_SIZE = (0 * sizeof(uint32_t));

namespace po = boost::program_options;

void check_arg_file_exists(po::variables_map &vm_in, std::string name) {
  if (!vm_in.count(name)) {
    throw std::runtime_error("Error: no " + name + " file was provided\n");
  } else {
    std::ifstream test(vm_in[name].as<std::string>());
    if (!test) {
      throw std::runtime_error("The " + name + " file " +
                               vm_in[name].as<std::string>() +
                               " does not exist.\n");
    }
  }
}

std::vector<uint32_t> load_instr_sequence(std::string instr_path) {
  std::ifstream instr_file(instr_path);
  std::string line;
  std::vector<uint32_t> instr_v;
  while (std::getline(instr_file, line)) {
    std::istringstream iss(line);
    uint32_t a;
    if (!(iss >> std::hex >> a)) {
      throw std::runtime_error("Unable to parse instruction file\n");
    }
    instr_v.push_back(a);
  }
  return instr_v;
}

template <typename T>
void conv_out_s2_nchw_fchw(std::vector<T> a, std::vector<T> b,
                           std::vector<T> &r) {
  int s = 2;
  for (size_t batch = 0; batch < BATCH; batch++) {
    for (size_t cout = 0; cout < CHOUT; cout++) {
      for (size_t y = 0; y < YOUT; y++) {
        for (size_t x = 0; x < XOUT; x++) {
          size_t idx =
              batch * CHOUT * XOUT * YOUT + y * XOUT * CHOUT + x * CHOUT + cout;
          r[idx] = (T)(0);
          for (size_t cin = 0; cin < CHIN; cin++) {
            for (size_t ky = 0; ky < K; ky++) {
              for (size_t kx = 0; kx < K; kx++) {
                T _a = a[batch * CHIN * XIN * YIN + (y * s + ky) * XIN * CHIN +
                         (x * s + kx) * CHIN + cin];
                T _b = b[ky * CHIN * CHOUT * K + kx * CHOUT * CHIN +
                         cin * CHOUT + cout];
                r[idx] += _a * _b;
              }
            }
          }
        }
      }
    }
  }
}

void write_out_trace(char *traceOutPtr, size_t trace_size, std::string path) {
  std::ofstream fout(path);
  uint32_t *traceOut = (uint32_t *)traceOutPtr;
  for (int i = 0; i < trace_size / sizeof(traceOut[0]); i++) {
    fout << std::setfill('0') << std::setw(8) << std::hex << (int)traceOut[i];
    fout << std::endl;
  }
}

int main(int argc, const char *argv[]) {

  // Program arguments parsing
  po::options_description desc("Allowed options");
  desc.add_options()("help,h", "produce help message")(
      "xclbin,x", po::value<std::string>()->required(),
      "the input xclbin path")(
      "kernel,k", po::value<std::string>()->required(),
      "the kernel name in the XCLBIN (for instance PP_PRE_FD)")(
      "verbosity,v", po::value<int>()->default_value(0),
      "the verbosity of the output")(
      "instr,i", po::value<std::string>()->required(),
      "path of file containing userspace instructions to be sent to the LX6")(
      "trace_sz,t", po::value<int>()->default_value(0),
      "size of trace buffer (in bytes)")(
      "trace_file", po::value<std::string>()->default_value("trace.txt"),
      "where to store trace output");
  po::variables_map vm;

  try {
    po::store(po::parse_command_line(argc, argv, desc), vm);
    po::notify(vm);

    if (vm.count("help")) {
      std::cout << desc << "\n";
      return 1;
    }
  } catch (const std::exception &ex) {
    std::cerr << ex.what() << "\n\n";
    std::cerr << "Usage:\n" << desc << "\n";
    return 1;
  }

  int trace_size = vm["trace_sz"].as<int>();

  check_arg_file_exists(vm, "xclbin");
  check_arg_file_exists(vm, "instr");

  std::vector<uint32_t> instr_v =
      load_instr_sequence(vm["instr"].as<std::string>());

  int verbosity = vm["verbosity"].as<int>();
  if (verbosity >= 1)
    std::cout << "Sequence instr count: " << instr_v.size() << "\n";

  // Start the XRT test code
  // Get a device handle
  unsigned int device_index = 0;
  auto device = xrt::device(device_index);

  // Load the xclbin
  if (verbosity >= 1)
    std::cout << "Loading xclbin: " << vm["xclbin"].as<std::string>() << "\n";
  auto xclbin = xrt::xclbin(vm["xclbin"].as<std::string>());

  if (verbosity >= 1)
    std::cout << "Kernel opcode: " << vm["kernel"].as<std::string>() << "\n";
  std::string Node = vm["kernel"].as<std::string>();

  // Get the kernel from the xclbin
  auto xkernels = xclbin.get_kernels();
  auto xkernel = *std::find_if(xkernels.begin(), xkernels.end(),
                               [Node](xrt::xclbin::kernel &k) {
                                 auto name = k.get_name();
                                 std::cout << "Name: " << name << std::endl;
                                 return name.rfind(Node, 0) == 0;
                               });
  auto kernelName = xkernel.get_name();

  if (verbosity >= 1)
    std::cout << "Registering xclbin: " << vm["xclbin"].as<std::string>()
              << "\n";

  device.register_xclbin(xclbin);

  // get a hardware context
  if (verbosity >= 1)
    std::cout << "Getting hardware context.\n";
  xrt::hw_context context(device, xclbin.get_uuid());

  // get a kernel handle
  if (verbosity >= 1)
    std::cout << "Getting handle to kernel:" << kernelName << "\n";
  auto kernel = xrt::kernel(context, kernelName);

  auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(int),
                          XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
  auto bo_a =
      xrt::bo(device, A_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
  auto bo_b =
      xrt::bo(device, B_SIZE, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
  auto bo_c = xrt::bo(device, C_SIZE + trace_size, XRT_BO_FLAGS_HOST_ONLY,
                      kernel.group_id(5));

  if (verbosity >= 1)
    std::cout << "Writing data into buffer objects.\n";
  A_DATATYPE *bufA = bo_a.map<A_DATATYPE *>();
  std::vector<A_DATATYPE> AVec;
  for (int i = 0; i < A_VOLUME; i++)
    AVec.push_back(rand() % UINT16_MAX);
  memcpy(bufA, AVec.data(), (AVec.size() * sizeof(A_DATATYPE)));
  B_DATATYPE *bufB = bo_b.map<B_DATATYPE *>();
  std::vector<B_DATATYPE> BVec;
  for (int i = 0; i < B_VOLUME; i++)
    BVec.push_back(rand() % UINT16_MAX);
  memcpy(bufB, BVec.data(), (BVec.size() * sizeof(B_DATATYPE)));
  C_DATATYPE *bufC = bo_c.map<C_DATATYPE *>();
  memset(bufC, 0, C_SIZE + trace_size);

  void *bufInstr = bo_instr.map<void *>();
  memcpy(bufInstr, instr_v.data(), instr_v.size() * sizeof(int));

  bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);
  bo_c.sync(XCL_BO_SYNC_BO_TO_DEVICE);

  if (verbosity >= 1)
    std::cout << "Running Kernel.\n";
  unsigned int opcode = 3;
  auto run = kernel(opcode, bo_instr, instr_v.size(), bo_a, bo_b, bo_c);
  run.wait();

  bo_c.sync(XCL_BO_SYNC_BO_FROM_DEVICE);

  C_DATATYPE *bufOut = bo_c.map<C_DATATYPE *>();

  int errors = 0;
  int max_errors = 100;

  std::vector<C_DATATYPE> output_ref0;
  for (uint32_t i = 0; i < C_VOLUME; i++)
    output_ref0.push_back(0);
  conv_out_s2_nchw_fchw(AVec, BVec, output_ref0);

  for (uint32_t i = 0; i < C_VOLUME; i++) {
    if (bufOut[i] != output_ref0[i]) {
      errors++;
      if (errors < max_errors) {
        std::cout << "\nerror, id " << i << " expected "
                  << std::to_string(output_ref0[i]) << ", got"
                  << std::to_string(bufOut[i]) << "\n";
      }
    }
  }

  if (trace_size > 0) {
    write_out_trace(((char *)bufC) + C_SIZE, trace_size,
                    vm["trace_file"].as<std::string>());
  }

  if (!errors) {
    std::cout << "\nPASS!\n\n";
    return 0;
  } else {
    std::cout << "\nerror count: " << errors << "\n\n";
    std::cout << "\nfailed.\n\n";
    return 1;
  }
}
