//===- utility.cpp ---------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022, Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
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
