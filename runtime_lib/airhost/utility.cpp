//===- utility.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
//
// Permission is hereby granted, free of charge, to any person obtaining a
// copy of this software and associated documentation files (the "Software"),
// to deal in the Software without restriction, including without limitation
// the rights to use, copy, modify, merge, publish, distribute, sublicense,
// and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
// OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
// DEALINGS IN THE SOFTWARE.
//
//===----------------------------------------------------------------------===//

/*
 * -*- coding: utf-8 -*-
 *
 *
 * (c) 2016 by the author(s)
 *
 * Author(s):
 *    Andre Richter, andre.richter@tum.de
 */

#include "utility.hpp"

#include <cstring>
#include <dirent.h>
#include <fstream> // ifstream
#include <iomanip> // setbase()

// Modified from
// https://github.com/andre-richter/easy-pci-mmap/blob/master/utility.cpp
void utility::get_pci_dbdf(std::vector<std::string> *bdf_vect, uint32_t vendor,
                           uint32_t device, unsigned int func_num) {
  DIR *dir;
  std::string err = std::string();

  if (bdf_vect == NULL) {
    printf("Passed NULL bdf_vect\n");
    return;
  }

  if ((dir = opendir("/sys/bus/pci/devices")) == nullptr) {
    printf("Cannot open directory /sys/bus/pci/devices\n");
    return;
  }

  // iterate over all PCIe devices
  struct dirent *d;
  std::ifstream ifstr;
  std::string bdf_found;

  while ((d = readdir(dir)) != nullptr) {

    // only consider actual device folders, not ./ and ../
    if (strstr(d->d_name, "0000:") != nullptr) {
      bdf_found = std::string(d->d_name);

      // Continue only if the function number matches
      if ((unsigned int)(bdf_found.back() - '0') != func_num)
        continue;

      std::string path("/sys/bus/pci/devices/" + bdf_found);

      // read vendor id
      ifstr.open(path + "/vendor");
      std::string tmp((std::istreambuf_iterator<char>(ifstr)),
                      std::istreambuf_iterator<char>());
      ifstr.close();

      // check if vendor id is correct
      if (std::stoul(tmp, nullptr, 16) == vendor) {

        // read device id
        ifstr.open(path + "/device");
        std::string tmp((std::istreambuf_iterator<char>(ifstr)),
                        std::istreambuf_iterator<char>());
        ifstr.close();

        // check if device also fits
        if (std::stoul(tmp, nullptr, 16) == device)
          bdf_vect->push_back("/sys/bus/pci/devices/" + bdf_found +
                              "/resource");
      }
    }
  }

  return;
}
