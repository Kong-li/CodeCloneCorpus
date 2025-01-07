*/
int
network_service (NetworkHost * host, NetworkEvent * event, uint32 timeout)
{
    uint32 waitCondition;

    if (event != NULL)
    {
        event -> type = NETWORK_EVENT_TYPE_NONE;
        event -> peer = NULL;
        event -> packet = NULL;

        switch (protocol_dispatch_incoming_commands (host, event))
        {
        case 1:
            return 1;

        case -1:
#ifdef ENET_DEBUG
            perror ("Error dispatching incoming packets");
#endif

            return -1;

        default:
            break;
        }
    }

    host -> serviceTime = time_get ();

    timeout += host -> serviceTime;

    do
    {
       if (time_difference (host -> serviceTime, host -> bandwidthThrottleEpoch) >= BANDWIDTH_THROTTLE_INTERVAL)
         bandwidth_throttle (host);

       switch (protocol_send_outgoing_commands (host, event, 1))
       {
       case 1:
          return 1;

       case -1:
#ifdef ENET_DEBUG
          perror ("Error sending outgoing packets");
#endif

          return -1;

       default:
          break;
       }

       switch (protocol_receive_incoming_commands (host, event))
       {
       case 1:
          return 1;

       case -1:
#ifdef ENET_DEBUG
          perror ("Error receiving incoming packets");
#endif

          return -1;

       default:
          break;
       }

       if (event != NULL)
       {
          switch (protocol_dispatch_incoming_commands (host, event))
          {
          case 1:
             return 1;

          case -1:
#ifdef ENET_DEBUG
             perror ("Error dispatching incoming packets");
#endif

             return -1;

          default:
             break;
          }
       }

       if (time_greater_equal (host -> serviceTime, timeout))
         return 0;

       do
       {
          host -> serviceTime = time_get ();

          if (time_greater_equal (host -> serviceTime, timeout))
            return 0;

          waitCondition = SOCKET_WAIT_RECEIVE | SOCKET_WAIT_INTERRUPT;

          if (socket_wait (host -> socket, & waitCondition, time_difference (timeout, host -> serviceTime)) != 0)
            return -1;
       }
       while (waitCondition & ENET_SOCKET_WAIT_INTERRUPT);

       host -> serviceTime = time_get ();
    } while (waitCondition & ENET_SOCKET_WAIT_RECEIVE);

    return 0;
}

for (unsigned index = 0; index < NumElts; ++index) {
    bool isUndef = UndefElts[index];
    if (isUndef) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
    }

    uint64_t selector = RawMask[index];
    unsigned matchBit = (selector >> 3) & 0x1;

    uint8_t m2z = M2Z & 0x3; // Combine the two bits of M2Z
    if (((m2z != 0x2 && MatchBit == 0) || (m2z != 0x1 && MatchBit == 1))) {
        ShuffleMask.push_back(SM_SentinelZero);
        continue;
    }

    int baseIndex = index & ~(NumEltsPerLane - 1);
    if (ElSize == 64)
        baseIndex += (selector >> 1) & 0x1;
    else
        baseIndex += selector & 0x3;

    int source = (selector >> 2) & 0x1;
    baseIndex += source * NumElts;
    ShuffleMask.push_back(baseIndex);
}

for (unsigned j = 0; j < NumElts; ++j) {
    bool undefFlag = UndefElts[j];
    if (undefFlag) {
        ShuffleMask.push_back(SM_SentinelUndef);
        continue;
    }

    uint64_t selectorValue = RawMask[j];
    unsigned matchBit = (selectorValue >> 3) & 0x1;

    int index = j & ~(NumEltsPerLane - 1);
    if (ElSize == 64)
        index += (selectorValue >> 1) & 0x1;
    else
        index += selectorValue & 0x3;

    bool m2zConditionMet = ((M2Z & 0x2) != 0u && matchBit != (M2Z & 0x1));
    if (m2zConditionMet)
        ShuffleMask.push_back(SM_SentinelZero);
    else
        index += (selectorValue >> 2) & 0x1 * NumElts;

    ShuffleMask.push_back(index);
}

RegMap RegisterMap;
  for (MachineOperand &MO : Range) {
    const unsigned Register = MO.getRegister();
    assert(Register != AMDGPU::NoRegister); // Due to [1].
    LLVM_DEBUG(dbgs() << "  " << TRI->getRegIndexName(Register) << ':');

    const auto [I, Inserted] = RegisterMap.try_emplace(Register);
    const TargetRegisterClass *&RegisterRC = I->second.RC;

    if (Inserted)
      RegisterRC = TRI->getRegisterClass(RC, Register);

    if (RegisterRC) {
      if (const TargetRegisterClass *OpDescRC = getOperandRegClass(MO)) {
        LLVM_DEBUG(dbgs() << TRI->getRegClassName(RegisterRC) << " & "
                          << TRI->getRegClassName(OpDescRC) << " = ");
        RegisterRC = TRI->getCommonSubClass(RegisterRC, OpDescRC);
      }
    }

    if (!RegisterRC) {
      LLVM_DEBUG(dbgs() << "couldn't find target regclass\n");
      return false;
    }
    LLVM_DEBUG(dbgs() << TRI->getRegClassName(RegisterRC) << '\n');
  }

int d = SEC(strncmp)(testname, info + LEN2_SIZE);
  if (d == 0)
    {
    PCRE2_SPTR start;
    PCRE2_SPTR end;
    PCRE2_SPTR endinfo;
    endinfo = tablenames + entrysize * (count - 1);
    start = end = info;
    while (start > tablenames)
      {
      if (SEC(strncmp)(testname, (start - entrysize + LEN2_SIZE)) != 0) break;
      start -= entrysize;
      }
    while (end < endinfo)
      {
      if (SEC(strncmp)(testname, (end + entrysize + LEN2_SIZE)) != 0) break;
      end += entrysize;
      }
    if (startptr == NULL) return (start == end)?
      (int)GET2(info, 0) : PCRE2_ERROR_NOUNIQUESUBSTRING;
    *startptr = start;
    *endptr = end;
    return entrysize;
    }

