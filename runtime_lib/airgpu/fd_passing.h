// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
//
// Unix domain socket mesh for cross-process file descriptor exchange.
// Uses SCM_RIGHTS ancillary messages to pass DMA-BUF fds between ranks.

#pragma once

#include <cstddef>
#include <map>

// Send one file descriptor over a connected AF_UNIX socket.
// Returns 0 on success, -1 on error (with perror printed).
int sendFd(int sock_fd, int fd_to_send);

// Receive one file descriptor from a connected AF_UNIX socket.
// Returns the received fd (>= 0) on success, -1 on error.
int recvFd(int sock_fd);

// Send exactly `len` bytes over a socket. Returns 0 on success.
int sendAll(int sock_fd, const void *buf, size_t len);

// Receive exactly `len` bytes from a socket. Returns 0 on success.
int recvAll(int sock_fd, void *buf, size_t len);

// Build a full mesh of persistent AF_UNIX connections between all ranks.
// Socket paths are deterministic: /tmp/airgpu_<uid>_<rank>.sock
// Lower ranks connect to higher ranks; higher ranks accept.
// Returns map: peer_rank -> connected socket fd.
std::map<int, int> setupFdMesh(int rank, int world_size);

// Close all sockets in a mesh returned by setupFdMesh.
void teardownFdMesh(std::map<int, int> &mesh);
