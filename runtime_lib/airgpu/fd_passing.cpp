// SPDX-License-Identifier: MIT
// Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.

#include "fd_passing.h"
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>
#include <unistd.h>

static std::string sockPath(int rank) {
  const char *job_id = std::getenv("AIRGPU_JOB_ID");
  char buf[108];
  if (job_id && job_id[0] != '\0')
    snprintf(buf, sizeof(buf), "/tmp/airgpu_%s_%d.sock", job_id, rank);
  else
    snprintf(buf, sizeof(buf), "/tmp/airgpu_%d_%d.sock",
             static_cast<int>(getuid()), rank);
  return std::string(buf);
}

int sendFd(int sock_fd, int fd_to_send) {
  char dummy = '\0';
  struct iovec iov = {};
  iov.iov_base = &dummy;
  iov.iov_len = 1;

  union {
    struct cmsghdr hdr;
    char buf[CMSG_SPACE(sizeof(int))];
  } cmsg_buf = {};

  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf.buf;
  msg.msg_controllen = sizeof(cmsg_buf.buf);

  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  cmsg->cmsg_level = SOL_SOCKET;
  cmsg->cmsg_type = SCM_RIGHTS;
  cmsg->cmsg_len = CMSG_LEN(sizeof(int));
  memcpy(CMSG_DATA(cmsg), &fd_to_send, sizeof(int));

  if (sendmsg(sock_fd, &msg, 0) < 0) {
    perror("airgpu: sendFd");
    return -1;
  }
  return 0;
}

int recvFd(int sock_fd) {
  char dummy;
  struct iovec iov = {};
  iov.iov_base = &dummy;
  iov.iov_len = 1;

  union {
    struct cmsghdr hdr;
    char buf[CMSG_SPACE(sizeof(int))];
  } cmsg_buf = {};

  struct msghdr msg = {};
  msg.msg_iov = &iov;
  msg.msg_iovlen = 1;
  msg.msg_control = cmsg_buf.buf;
  msg.msg_controllen = sizeof(cmsg_buf.buf);

  if (recvmsg(sock_fd, &msg, 0) < 0) {
    perror("airgpu: recvFd");
    return -1;
  }

  struct cmsghdr *cmsg = CMSG_FIRSTHDR(&msg);
  if (cmsg && cmsg->cmsg_level == SOL_SOCKET && cmsg->cmsg_type == SCM_RIGHTS) {
    int fd;
    memcpy(&fd, CMSG_DATA(cmsg), sizeof(int));
    return fd;
  }

  fprintf(stderr, "airgpu: recvFd: no SCM_RIGHTS in message\n");
  return -1;
}

int sendAll(int sock_fd, const void *buf, size_t len) {
  const char *p = static_cast<const char *>(buf);
  while (len > 0) {
    ssize_t n = send(sock_fd, p, len, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      perror("airgpu: sendAll");
      return -1;
    }
    p += n;
    len -= n;
  }
  return 0;
}

int recvAll(int sock_fd, void *buf, size_t len) {
  char *p = static_cast<char *>(buf);
  while (len > 0) {
    ssize_t n = recv(sock_fd, p, len, 0);
    if (n < 0) {
      if (errno == EINTR)
        continue;
      perror("airgpu: recvAll");
      return -1;
    }
    if (n == 0) {
      fprintf(stderr, "airgpu: recvAll: peer disconnected\n");
      return -1;
    }
    p += n;
    len -= n;
  }
  return 0;
}

std::map<int, int> setupFdMesh(int rank, int world_size) {
  std::map<int, int> conns;
  if (world_size <= 1)
    return conns;

  // Create listener
  std::string my_path = sockPath(rank);
  unlink(my_path.c_str());

  int listener = socket(AF_UNIX, SOCK_STREAM, 0);
  if (listener < 0) {
    perror("airgpu: socket");
    abort();
  }

  struct sockaddr_un addr = {};
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, my_path.c_str(), sizeof(addr.sun_path) - 1);

  if (bind(listener, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)) <
      0) {
    perror("airgpu: bind");
    abort();
  }
  if (listen(listener, world_size) < 0) {
    perror("airgpu: listen");
    abort();
  }

  // Connect to all lower ranks
  for (int peer = 0; peer < rank; peer++) {
    int s = socket(AF_UNIX, SOCK_STREAM, 0);
    if (s < 0) {
      perror("airgpu: socket");
      abort();
    }

    std::string peer_path = sockPath(peer);
    struct sockaddr_un peer_addr = {};
    peer_addr.sun_family = AF_UNIX;
    strncpy(peer_addr.sun_path, peer_path.c_str(),
            sizeof(peer_addr.sun_path) - 1);

    // Retry with timeout — peer may not have bound yet
    int connected = 0;
    for (int attempt = 0; attempt < 1000; attempt++) {
      if (connect(s, reinterpret_cast<struct sockaddr *>(&peer_addr),
                  sizeof(peer_addr)) == 0) {
        connected = 1;
        break;
      }
      usleep(10000); // 10ms
    }
    if (!connected) {
      fprintf(stderr, "airgpu: rank %d timed out connecting to rank %d at %s\n",
              rank, peer, peer_path.c_str());
      abort();
    }

    // Identify ourselves
    uint32_t my_rank = static_cast<uint32_t>(rank);
    sendAll(s, &my_rank, sizeof(my_rank));
    conns[peer] = s;
  }

  // Accept connections from higher ranks
  for (int i = rank + 1; i < world_size; i++) {
    int client = accept(listener, nullptr, nullptr);
    if (client < 0) {
      perror("airgpu: accept");
      abort();
    }
    uint32_t peer_rank = 0;
    recvAll(client, &peer_rank, sizeof(peer_rank));
    conns[static_cast<int>(peer_rank)] = client;
  }

  close(listener);
  unlink(my_path.c_str());

  fprintf(stderr, "airgpu: rank %d fd mesh established with %zu peers\n", rank,
          conns.size());
  return conns;
}

void teardownFdMesh(std::map<int, int> &mesh) {
  for (auto &[peer, fd] : mesh)
    close(fd);
  mesh.clear();
}
