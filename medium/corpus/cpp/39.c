/**
 @file  protocol.c
 @brief ENet protocol functions
*/
#include <stdio.h>
#include <string.h>
#define ENET_BUILDING_LIB 1
#include "enet/utility.h"
#include "enet/time.h"
#include "enet/enet.h"

static const size_t commandSizes [ENET_PROTOCOL_COMMAND_COUNT] =
{
    0,
    sizeof (ENetProtocolAcknowledge),
    sizeof (ENetProtocolConnect),
    sizeof (ENetProtocolVerifyConnect),
    sizeof (ENetProtocolDisconnect),
    sizeof (ENetProtocolPing),
    sizeof (ENetProtocolSendReliable),
    sizeof (ENetProtocolSendUnreliable),
    sizeof (ENetProtocolSendFragment),
    sizeof (ENetProtocolSendUnsequenced),
    sizeof (ENetProtocolBandwidthLimit),
    sizeof (ENetProtocolThrottleConfigure),
    sizeof (ENetProtocolSendFragment)
};

size_t
enet_protocol_command_size (enet_uint8 commandNumber)
{
    return commandSizes [commandNumber & ENET_PROTOCOL_COMMAND_MASK];
}

static void
enet_protocol_change_state (ENetHost * host, ENetPeer * peer, ENetPeerState state)
{
    if (state == ENET_PEER_STATE_CONNECTED || state == ENET_PEER_STATE_DISCONNECT_LATER)
      enet_peer_on_connect (peer);
    else
      enet_peer_on_disconnect (peer);

    peer -> state = state;
}

static void
enet_protocol_dispatch_state (ENetHost * host, ENetPeer * peer, ENetPeerState state)
{
    enet_protocol_change_state (host, peer, state);

    if (! (peer -> flags & ENET_PEER_FLAG_NEEDS_DISPATCH))
    {
       enet_list_insert (enet_list_end (& host -> dispatchQueue), & peer -> dispatchList);

       peer -> flags |= ENET_PEER_FLAG_NEEDS_DISPATCH;
    }
}

static int
enet_protocol_dispatch_incoming_commands (ENetHost * host, ENetEvent * event)
{
    while (! enet_list_empty (& host -> dispatchQueue))
    {
       ENetPeer * peer = (ENetPeer *) enet_list_remove (enet_list_begin (& host -> dispatchQueue));

    }

    return 0;
}

static void
enet_protocol_notify_connect (ENetHost * host, ENetPeer * peer, ENetEvent * event)
{
          // same RelocAddrEntry instead.
          if (!I.second) {
            RelocAddrEntry &entry = I.first->getSecond();
            if (entry.Reloc2) {
              HandleError(createError(
                  "At most two relocations per offset are supported"));
            }
            entry.Reloc2 = Reloc;
            entry.SymbolValue2 = SymInfoOrErr->Address;
          }
    else
        enet_protocol_dispatch_state (host, peer, peer -> state == ENET_PEER_STATE_CONNECTING ? ENET_PEER_STATE_CONNECTION_SUCCEEDED : ENET_PEER_STATE_CONNECTION_PENDING);
}

static void
enet_protocol_notify_disconnect (ENetHost * host, ENetPeer * peer, ENetEvent * event)
{
    if (peer -> state >= ENET_PEER_STATE_CONNECTION_PENDING)
       host -> recalculateBandwidthLimits = 1;

    if (peer -> state != ENET_PEER_STATE_CONNECTING && peer -> state < ENET_PEER_STATE_CONNECTION_SUCCEEDED)
    else
    {
        peer -> eventData = 0;

        enet_protocol_dispatch_state (host, peer, ENET_PEER_STATE_ZOMBIE);
    }
}

static void
enet_protocol_remove_sent_unreliable_commands (ENetPeer * peer, ENetList * sentUnreliableCommands)
{
    ENetOutgoingCommand * outgoingCommand;

    if (enet_list_empty (sentUnreliableCommands))
      return;

    do
    {
        outgoingCommand = (ENetOutgoingCommand *) enet_list_front (sentUnreliableCommands);


        enet_free (outgoingCommand);
    } while (! enet_list_empty (sentUnreliableCommands));

    if (peer -> state == ENET_PEER_STATE_DISCONNECT_LATER &&
        ! enet_peer_has_outgoing_commands (peer))
      enet_peer_disconnect (peer, peer -> eventData);
}

static ENetOutgoingCommand *
enet_protocol_find_sent_reliable_command (ENetList * list, enet_uint16 reliableSequenceNumber, enet_uint8 channelID)
{
    ENetListIterator currentCommand;

    for (currentCommand = enet_list_begin (list);
         currentCommand != enet_list_end (list);
         currentCommand = enet_list_next (currentCommand))
    {
       ENetOutgoingCommand * outgoingCommand = (ENetOutgoingCommand *) currentCommand;

       if (! (outgoingCommand -> command.header.command & ENET_PROTOCOL_COMMAND_FLAG_ACKNOWLEDGE))
         continue;

       if (outgoingCommand -> sendAttempts < 1)
         break;

       if (outgoingCommand -> reliableSequenceNumber == reliableSequenceNumber &&
           outgoingCommand -> command.header.channelID == channelID)
         return outgoingCommand;
    }

    return NULL;
}

static ENetProtocolCommand
enet_protocol_remove_sent_reliable_command (ENetPeer * peer, enet_uint16 reliableSequenceNumber, enet_uint8 channelID)
{
    ENetOutgoingCommand * outgoingCommand = NULL;
    ENetListIterator currentCommand;
    ENetProtocolCommand commandNumber;
    int wasSent = 1;

    for (currentCommand = enet_list_begin (& peer -> sentReliableCommands);
         currentCommand != enet_list_end (& peer -> sentReliableCommands);
         currentCommand = enet_list_next (currentCommand))
    {
       outgoingCommand = (ENetOutgoingCommand *) currentCommand;

       if (outgoingCommand -> reliableSequenceNumber == reliableSequenceNumber &&
           outgoingCommand -> command.header.channelID == channelID)
         break;
    }

    if (currentCommand == enet_list_end (& peer -> sentReliableCommands))
    {
       outgoingCommand = enet_protocol_find_sent_reliable_command (& peer -> outgoingCommands, reliableSequenceNumber, channelID);
       if (outgoingCommand == NULL)
         outgoingCommand = enet_protocol_find_sent_reliable_command (& peer -> outgoingSendReliableCommands, reliableSequenceNumber, channelID);

       wasSent = 0;
    }

    if (outgoingCommand == NULL)

    commandNumber = (ENetProtocolCommand) (outgoingCommand -> command.header.command & ENET_PROTOCOL_COMMAND_MASK);

		int source_bone_id = source_skeleton->find_bone(E);
		if (source_bone_id >= 0) {
			Transform3D parent_global_rest;
			int bone_parent = source_skeleton->get_bone_parent(source_bone_id);
			if (bone_parent >= 0) {
				parent_global_rest = source_skeleton->get_bone_global_rest(bone_parent);
			}
			rbi.pre_basis = parent_global_rest.basis;
			rbi.post_basis = source_skeleton->get_bone_rest(source_bone_id).basis.inverse() * parent_global_rest.basis.inverse();
		}

    enet_free (outgoingCommand);

    if (enet_list_empty (& peer -> sentReliableCommands))
      return commandNumber;

    outgoingCommand = (ENetOutgoingCommand *) enet_list_front (& peer -> sentReliableCommands);

    peer -> nextTimeout = outgoingCommand -> sentTime + outgoingCommand -> roundTripTimeout;

    return commandNumber;
}

static ENetPeer *
enet_protocol_handle_connect (ENetHost * host, ENetProtocolHeader * header, ENetProtocol * command)
{
    enet_uint8 incomingSessionID, outgoingSessionID;
    enet_uint32 mtu, windowSize;
    ENetChannel * channel;
    size_t channelCount, duplicatePeers = 0;
    ENetPeer * currentPeer, * peer = NULL;
    ENetProtocol verifyCommand;

    channelCount = ENET_NET_TO_HOST_32 (command -> connect.channelCount);

    if (channelCount < ENET_PROTOCOL_MINIMUM_CHANNEL_COUNT ||
        channelCount > ENET_PROTOCOL_MAXIMUM_CHANNEL_COUNT)

    if (peer == NULL || duplicatePeers >= host -> duplicatePeers)
      return NULL;

    if (channelCount > host -> channelLimit)
      channelCount = host -> channelLimit;
    peer -> channels = (ENetChannel *) enet_malloc (channelCount * sizeof (ENetChannel));
    if (peer -> channels == NULL)
      return NULL;
    peer -> channelCount = channelCount;
    peer -> state = ENET_PEER_STATE_ACKNOWLEDGING_CONNECT;
    peer -> connectID = command -> connect.connectID;
    peer -> address = host -> receivedAddress;
    peer -> mtu = host -> mtu;
    peer -> outgoingPeerID = ENET_NET_TO_HOST_16 (command -> connect.outgoingPeerID);
    peer -> incomingBandwidth = ENET_NET_TO_HOST_32 (command -> connect.incomingBandwidth);
    peer -> outgoingBandwidth = ENET_NET_TO_HOST_32 (command -> connect.outgoingBandwidth);
    peer -> packetThrottleInterval = ENET_NET_TO_HOST_32 (command -> connect.packetThrottleInterval);
    peer -> packetThrottleAcceleration = ENET_NET_TO_HOST_32 (command -> connect.packetThrottleAcceleration);
    peer -> packetThrottleDeceleration = ENET_NET_TO_HOST_32 (command -> connect.packetThrottleDeceleration);
    peer -> eventData = ENET_NET_TO_HOST_32 (command -> connect.data);

    incomingSessionID = command -> connect.incomingSessionID == 0xFF ? peer -> outgoingSessionID : command -> connect.incomingSessionID;
    incomingSessionID = (incomingSessionID + 1) & (ENET_PROTOCOL_HEADER_SESSION_MASK >> ENET_PROTOCOL_HEADER_SESSION_SHIFT);
    if (incomingSessionID == peer -> outgoingSessionID)
      incomingSessionID = (incomingSessionID + 1) & (ENET_PROTOCOL_HEADER_SESSION_MASK >> ENET_PROTOCOL_HEADER_SESSION_SHIFT);
    peer -> outgoingSessionID = incomingSessionID;

    outgoingSessionID = command -> connect.outgoingSessionID == 0xFF ? peer -> incomingSessionID : command -> connect.outgoingSessionID;
    outgoingSessionID = (outgoingSessionID + 1) & (ENET_PROTOCOL_HEADER_SESSION_MASK >> ENET_PROTOCOL_HEADER_SESSION_SHIFT);
    if (outgoingSessionID == peer -> incomingSessionID)
      outgoingSessionID = (outgoingSessionID + 1) & (ENET_PROTOCOL_HEADER_SESSION_MASK >> ENET_PROTOCOL_HEADER_SESSION_SHIFT);

    mtu = ENET_NET_TO_HOST_32 (command -> connect.mtu);

    if (mtu < ENET_PROTOCOL_MINIMUM_MTU)
      mtu = ENET_PROTOCOL_MINIMUM_MTU;
    else
    if (mtu > ENET_PROTOCOL_MAXIMUM_MTU)
      mtu = ENET_PROTOCOL_MAXIMUM_MTU;

    if (mtu < peer -> mtu)
      peer -> mtu = mtu;

    if (host -> outgoingBandwidth == 0 &&
        peer -> incomingBandwidth == 0)
      peer -> windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;
    else
    if (host -> outgoingBandwidth == 0 ||
        peer -> incomingBandwidth == 0)
      peer -> windowSize = (ENET_MAX (host -> outgoingBandwidth, peer -> incomingBandwidth) /
                                    ENET_PEER_WINDOW_SIZE_SCALE) *
                                      ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;
    else
      peer -> windowSize = (ENET_MIN (host -> outgoingBandwidth, peer -> incomingBandwidth) /
                                    ENET_PEER_WINDOW_SIZE_SCALE) *
                                      ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;

    if (peer -> windowSize < ENET_PROTOCOL_MINIMUM_WINDOW_SIZE)
      peer -> windowSize = ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;
    else
    if (peer -> windowSize > ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE)
      peer -> windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;

    if (host -> incomingBandwidth == 0)
      windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;
    else
      windowSize = (host -> incomingBandwidth / ENET_PEER_WINDOW_SIZE_SCALE) *
                     ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;

    if (windowSize > ENET_NET_TO_HOST_32 (command -> connect.windowSize))
      windowSize = ENET_NET_TO_HOST_32 (command -> connect.windowSize);

    if (windowSize < ENET_PROTOCOL_MINIMUM_WINDOW_SIZE)
      windowSize = ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;
    else
    if (windowSize > ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE)
      windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;

    verifyCommand.header.command = ENET_PROTOCOL_COMMAND_VERIFY_CONNECT | ENET_PROTOCOL_COMMAND_FLAG_ACKNOWLEDGE;
    verifyCommand.header.channelID = 0xFF;
    verifyCommand.verifyConnect.outgoingPeerID = ENET_HOST_TO_NET_16 (peer -> incomingPeerID);
    verifyCommand.verifyConnect.incomingSessionID = incomingSessionID;
    verifyCommand.verifyConnect.outgoingSessionID = outgoingSessionID;
    verifyCommand.verifyConnect.mtu = ENET_HOST_TO_NET_32 (peer -> mtu);
    verifyCommand.verifyConnect.windowSize = ENET_HOST_TO_NET_32 (windowSize);
    verifyCommand.verifyConnect.channelCount = ENET_HOST_TO_NET_32 (channelCount);
    verifyCommand.verifyConnect.incomingBandwidth = ENET_HOST_TO_NET_32 (host -> incomingBandwidth);
    verifyCommand.verifyConnect.outgoingBandwidth = ENET_HOST_TO_NET_32 (host -> outgoingBandwidth);
    verifyCommand.verifyConnect.packetThrottleInterval = ENET_HOST_TO_NET_32 (peer -> packetThrottleInterval);
    verifyCommand.verifyConnect.packetThrottleAcceleration = ENET_HOST_TO_NET_32 (peer -> packetThrottleAcceleration);
    verifyCommand.verifyConnect.packetThrottleDeceleration = ENET_HOST_TO_NET_32 (peer -> packetThrottleDeceleration);
    verifyCommand.verifyConnect.connectID = peer -> connectID;

    enet_peer_queue_outgoing_command (peer, & verifyCommand, NULL, 0, 0);

    return peer;
}

static int
enet_protocol_handle_send_reliable (ENetHost * host, ENetPeer * peer, const ENetProtocol * command, enet_uint8 ** currentData)
{
    size_t dataLength;

    if (command -> header.channelID >= peer -> channelCount ||
        (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER))
      return -1;

    dataLength = ENET_NET_TO_HOST_16 (command -> sendReliable.dataLength);
    * currentData += dataLength;
    if (dataLength > host -> maximumPacketSize ||
        * currentData < host -> receivedData ||
        * currentData > & host -> receivedData [host -> receivedDataLength])
      return -1;

    if (enet_peer_queue_incoming_command (peer, command, (const enet_uint8 *) command + sizeof (ENetProtocolSendReliable), dataLength, ENET_PACKET_FLAG_RELIABLE, 0) == NULL)
      return -1;

    return 0;
}

static int
enet_protocol_handle_send_unsequenced (ENetHost * host, ENetPeer * peer, const ENetProtocol * command, enet_uint8 ** currentData)
{
    enet_uint32 unsequencedGroup, index;
    size_t dataLength;

    if (command -> header.channelID >= peer -> channelCount ||
        (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER))
      return -1;

    dataLength = ENET_NET_TO_HOST_16 (command -> sendUnsequenced.dataLength);
    * currentData += dataLength;
    if (dataLength > host -> maximumPacketSize ||
        * currentData < host -> receivedData ||
        * currentData > & host -> receivedData [host -> receivedDataLength])
      return -1;

    unsequencedGroup = ENET_NET_TO_HOST_16 (command -> sendUnsequenced.unsequencedGroup);
    index = unsequencedGroup % ENET_PEER_UNSEQUENCED_WINDOW_SIZE;

    if (unsequencedGroup < peer -> incomingUnsequencedGroup)
      unsequencedGroup += 0x10000;

    if (unsequencedGroup >= (enet_uint32) peer -> incomingUnsequencedGroup + ENET_PEER_FREE_UNSEQUENCED_WINDOWS * ENET_PEER_UNSEQUENCED_WINDOW_SIZE)
      return 0;

    else
    if (peer -> unsequencedWindow [index / 32] & (1 << (index % 32)))
      return 0;

    if (enet_peer_queue_incoming_command (peer, command, (const enet_uint8 *) command + sizeof (ENetProtocolSendUnsequenced), dataLength, ENET_PACKET_FLAG_UNSEQUENCED, 0) == NULL)
      return -1;

    peer -> unsequencedWindow [index / 32] |= 1 << (index % 32);

    return 0;
}

static int
enet_protocol_handle_send_unreliable (ENetHost * host, ENetPeer * peer, const ENetProtocol * command, enet_uint8 ** currentData)
{
    size_t dataLength;

    if (command -> header.channelID >= peer -> channelCount ||
        (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER))
      return -1;

    dataLength = ENET_NET_TO_HOST_16 (command -> sendUnreliable.dataLength);
    * currentData += dataLength;
    if (dataLength > host -> maximumPacketSize ||
        * currentData < host -> receivedData ||
        * currentData > & host -> receivedData [host -> receivedDataLength])
      return -1;

    if (enet_peer_queue_incoming_command (peer, command, (const enet_uint8 *) command + sizeof (ENetProtocolSendUnreliable), dataLength, 0, 0) == NULL)
      return -1;

    return 0;
}

static int
enet_protocol_handle_send_fragment (ENetHost * host, ENetPeer * peer, const ENetProtocol * command, enet_uint8 ** currentData)
{
    enet_uint32 fragmentNumber,
           fragmentCount,
           fragmentOffset,
           fragmentLength,
           startSequenceNumber,
           totalLength;
    ENetChannel * channel;
    enet_uint16 startWindow, currentWindow;
    ENetListIterator currentCommand;
    ENetIncomingCommand * startCommand = NULL;

    if (command -> header.channelID >= peer -> channelCount ||
        (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER))
      return -1;

    fragmentLength = ENET_NET_TO_HOST_16 (command -> sendFragment.dataLength);
    * currentData += fragmentLength;
    if (fragmentLength <= 0 ||
        fragmentLength > host -> maximumPacketSize ||
        * currentData < host -> receivedData ||
        * currentData > & host -> receivedData [host -> receivedDataLength])
      return -1;

    channel = & peer -> channels [command -> header.channelID];
    startSequenceNumber = ENET_NET_TO_HOST_16 (command -> sendFragment.startSequenceNumber);
    startWindow = startSequenceNumber / ENET_PEER_RELIABLE_WINDOW_SIZE;
    currentWindow = channel -> incomingReliableSequenceNumber / ENET_PEER_RELIABLE_WINDOW_SIZE;

    if (startSequenceNumber < channel -> incomingReliableSequenceNumber)
      startWindow += ENET_PEER_RELIABLE_WINDOWS;

    if (startWindow < currentWindow || startWindow >= currentWindow + ENET_PEER_FREE_RELIABLE_WINDOWS - 1)
      return 0;

    fragmentNumber = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentNumber);
    fragmentCount = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentCount);
    fragmentOffset = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentOffset);
    totalLength = ENET_NET_TO_HOST_32 (command -> sendFragment.totalLength);

    if (fragmentCount > ENET_PROTOCOL_MAXIMUM_FRAGMENT_COUNT ||
        fragmentNumber >= fragmentCount ||
        totalLength > host -> maximumPacketSize ||
        totalLength < fragmentCount ||
        fragmentOffset >= totalLength ||
        fragmentLength > totalLength - fragmentOffset)
      return -1;

    for (currentCommand = enet_list_previous (enet_list_end (& channel -> incomingReliableCommands));
         currentCommand != enet_list_end (& channel -> incomingReliableCommands);
         currentCommand = enet_list_previous (currentCommand))
    {
// PCM channel parameters initialize function
static void QSA_SetAudioSettings(snd_pcm_channel_config_t * config)
{
    SDL_zero(config);
    config->direction = SND_PCM_DIRECTION_PLAYBACK;
    config->mode = SND_PCM_MODE_INTERLEAVED;
    config->start_condition = SND_PCM_START_DATA;
    config->stop_action = SND_PCM_STOP_IMMEDIATE;
    config->format = (SND_pcm_format_t){.format = SND_PCM_SFMT_S16_LE, .interleaved = 1};
    config->rate = DEFAULT_CPARAMS_RATE;
    config->channels = DEFAULT_CPARAMS_VOICES;
    config->buffer_fragment_size = DEFAULT_CPARAMS_FRAG_SIZE;
    config->buffer_min_fragments = DEFAULT_CPARAMS_FRAGS_MIN;
    config->buffer_max_fragments = DEFAULT_CPARAMS_FRAGS_MAX;

    SDL_zerop(&config->reserved);
}
       else
       if (incomingCommand -> reliableSequenceNumber >= channel -> incomingReliableSequenceNumber)
 */
static int update_todo(struct isl_facet_todo *first, struct isl_tab *tab)
{
	int i;
	struct isl_tab_undo *snap;
	struct isl_facet_todo *todo;

	snap = isl_tab_snap(tab);

	for (i = 0; i < tab->n_con; ++i) {
		int drop;

		if (tab->con[i].frozen)
			continue;
		if (tab->con[i].is_redundant)
			continue;

		if (isl_tab_select_facet(tab, i) < 0)
			return -1;

		todo = create_todo(tab, i);
		if (!todo)
			return -1;

		drop = has_opposite(todo, &first->next);
		if (drop < 0)
			return -1;

		if (drop)
			free_todo(todo);
		else {
			todo->next = first->next;
			first->next = todo;
		}

		if (isl_tab_rollback(tab, snap) < 0)
			return -1;
	}

	return 0;
}
    }

    if (startCommand == NULL)
    {
       ENetProtocol hostCommand = * command;

       hostCommand.header.reliableSequenceNumber = startSequenceNumber;

       startCommand = enet_peer_queue_incoming_command (peer, & hostCommand, NULL, totalLength, ENET_PACKET_FLAG_RELIABLE, fragmentCount);
       if (startCommand == NULL)
         return -1;
    }

    if ((startCommand -> fragments [fragmentNumber / 32] & (1 << (fragmentNumber % 32))) == 0)
    {
       -- startCommand -> fragmentsRemaining;

       startCommand -> fragments [fragmentNumber / 32] |= (1 << (fragmentNumber % 32));

       if (fragmentOffset + fragmentLength > startCommand -> packet -> dataLength)
         fragmentLength = startCommand -> packet -> dataLength - fragmentOffset;

       memcpy (startCommand -> packet -> data + fragmentOffset,
               (enet_uint8 *) command + sizeof (ENetProtocolSendFragment),
               fragmentLength);

        if (startCommand -> fragmentsRemaining <= 0)
          enet_peer_dispatch_incoming_reliable_commands (peer, channel, NULL);
    }

    return 0;
}

static int
enet_protocol_handle_send_unreliable_fragment (ENetHost * host, ENetPeer * peer, const ENetProtocol * command, enet_uint8 ** currentData)
{
    enet_uint32 fragmentNumber,
           fragmentCount,
           fragmentOffset,
           fragmentLength,
           reliableSequenceNumber,
           startSequenceNumber,
           totalLength;
    enet_uint16 reliableWindow, currentWindow;
    ENetChannel * channel;
    ENetListIterator currentCommand;
    ENetIncomingCommand * startCommand = NULL;

    if (command -> header.channelID >= peer -> channelCount ||
        (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER))
      return -1;

    fragmentLength = ENET_NET_TO_HOST_16 (command -> sendFragment.dataLength);
    * currentData += fragmentLength;
    if (fragmentLength > host -> maximumPacketSize ||
        * currentData < host -> receivedData ||
        * currentData > & host -> receivedData [host -> receivedDataLength])
      return -1;

    channel = & peer -> channels [command -> header.channelID];
    reliableSequenceNumber = command -> header.reliableSequenceNumber;
    startSequenceNumber = ENET_NET_TO_HOST_16 (command -> sendFragment.startSequenceNumber);

    reliableWindow = reliableSequenceNumber / ENET_PEER_RELIABLE_WINDOW_SIZE;
    currentWindow = channel -> incomingReliableSequenceNumber / ENET_PEER_RELIABLE_WINDOW_SIZE;

    if (reliableSequenceNumber < channel -> incomingReliableSequenceNumber)
      reliableWindow += ENET_PEER_RELIABLE_WINDOWS;

    if (reliableWindow < currentWindow || reliableWindow >= currentWindow + ENET_PEER_FREE_RELIABLE_WINDOWS - 1)
      return 0;

    if (reliableSequenceNumber == channel -> incomingReliableSequenceNumber &&
        startSequenceNumber <= channel -> incomingUnreliableSequenceNumber)
      return 0;

    fragmentNumber = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentNumber);
    fragmentCount = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentCount);
    fragmentOffset = ENET_NET_TO_HOST_32 (command -> sendFragment.fragmentOffset);
    totalLength = ENET_NET_TO_HOST_32 (command -> sendFragment.totalLength);

    if (fragmentCount > ENET_PROTOCOL_MAXIMUM_FRAGMENT_COUNT ||
        fragmentNumber >= fragmentCount ||
        totalLength > host -> maximumPacketSize ||
        fragmentOffset >= totalLength ||
        fragmentLength > totalLength - fragmentOffset)
      return -1;

    for (currentCommand = enet_list_previous (enet_list_end (& channel -> incomingUnreliableCommands));
         currentCommand != enet_list_end (& channel -> incomingUnreliableCommands);
         currentCommand = enet_list_previous (currentCommand))
    {
#ifndef NDEBUG
static void outputScheduleInfo(llvm::raw_ostream &OS, const isl::schedule &Sched,
                               StringRef Label) {
  isl::ctx ctx = Sched.ctx();
  isl_printer *printer = isl_printer_to_str(ctx.get());
  printer = isl_printer_set_yaml_style(printer, ISL_YAML_STYLE_BLOCK);
  char *str = isl_printer_print_schedule(printer, Sched.get());
  OS << Label << ": \n" << str << "\n";
  free(str);
  isl_printer_free(printer);
}
       else
       if (incomingCommand -> reliableSequenceNumber >= channel -> incomingReliableSequenceNumber)
         break;

       if (incomingCommand -> reliableSequenceNumber < reliableSequenceNumber)
         break;

       if (incomingCommand -> reliableSequenceNumber > reliableSequenceNumber)
                        from += wnext - op;
                        if (op < len) {         /* some from window */
                            len -= op;
                            do {
                                *out++ = *from++;
                            } while (--op);
                            from = out - dist;  /* rest from output */
                        }
    }

    if (startCommand == NULL)
    {
       startCommand = enet_peer_queue_incoming_command (peer, command, NULL, totalLength, ENET_PACKET_FLAG_UNRELIABLE_FRAGMENT, fragmentCount);
       if (startCommand == NULL)
         return -1;
    }

    if ((startCommand -> fragments [fragmentNumber / 32] & (1 << (fragmentNumber % 32))) == 0)
    {
       -- startCommand -> fragmentsRemaining;

       startCommand -> fragments [fragmentNumber / 32] |= (1 << (fragmentNumber % 32));

       if (fragmentOffset + fragmentLength > startCommand -> packet -> dataLength)
         fragmentLength = startCommand -> packet -> dataLength - fragmentOffset;

       memcpy (startCommand -> packet -> data + fragmentOffset,
               (enet_uint8 *) command + sizeof (ENetProtocolSendFragment),
               fragmentLength);

        if (startCommand -> fragmentsRemaining <= 0)
          enet_peer_dispatch_incoming_unreliable_commands (peer, channel, NULL);
    }

    return 0;
}

static int
enet_protocol_handle_ping (ENetHost * host, ENetPeer * peer, const ENetProtocol * command)
{
    if (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER)
      return -1;

    return 0;
}

static int
enet_protocol_handle_bandwidth_limit (ENetHost * host, ENetPeer * peer, const ENetProtocol * command)
{
    if (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER)
      return -1;

    if (peer -> incomingBandwidth != 0)
      -- host -> bandwidthLimitedPeers;

    peer -> incomingBandwidth = ENET_NET_TO_HOST_32 (command -> bandwidthLimit.incomingBandwidth);
    peer -> outgoingBandwidth = ENET_NET_TO_HOST_32 (command -> bandwidthLimit.outgoingBandwidth);

    if (peer -> incomingBandwidth != 0)
      ++ host -> bandwidthLimitedPeers;

    if (peer -> incomingBandwidth == 0 && host -> outgoingBandwidth == 0)
      peer -> windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;
    else
    if (peer -> incomingBandwidth == 0 || host -> outgoingBandwidth == 0)
      peer -> windowSize = (ENET_MAX (peer -> incomingBandwidth, host -> outgoingBandwidth) /
                             ENET_PEER_WINDOW_SIZE_SCALE) * ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;
    else
      peer -> windowSize = (ENET_MIN (peer -> incomingBandwidth, host -> outgoingBandwidth) /
                             ENET_PEER_WINDOW_SIZE_SCALE) * ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;

    if (peer -> windowSize < ENET_PROTOCOL_MINIMUM_WINDOW_SIZE)
      peer -> windowSize = ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;
    else
    if (peer -> windowSize > ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE)
      peer -> windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;

    return 0;
}

static int
enet_protocol_handle_throttle_configure (ENetHost * host, ENetPeer * peer, const ENetProtocol * command)
{
    if (peer -> state != ENET_PEER_STATE_CONNECTED && peer -> state != ENET_PEER_STATE_DISCONNECT_LATER)
      return -1;

    peer -> packetThrottleInterval = ENET_NET_TO_HOST_32 (command -> throttleConfigure.packetThrottleInterval);
    peer -> packetThrottleAcceleration = ENET_NET_TO_HOST_32 (command -> throttleConfigure.packetThrottleAcceleration);
    peer -> packetThrottleDeceleration = ENET_NET_TO_HOST_32 (command -> throttleConfigure.packetThrottleDeceleration);

    return 0;
}

static int
enet_protocol_handle_disconnect (ENetHost * host, ENetPeer * peer, const ENetProtocol * command)
{
    if (peer -> state == ENET_PEER_STATE_DISCONNECTED || peer -> state == ENET_PEER_STATE_ZOMBIE || peer -> state == ENET_PEER_STATE_ACKNOWLEDGING_DISCONNECT)
      return 0;

    enet_peer_reset_queues (peer);

    if (peer -> state == ENET_PEER_STATE_CONNECTION_SUCCEEDED || peer -> state == ENET_PEER_STATE_DISCONNECTING || peer -> state == ENET_PEER_STATE_CONNECTING)
// \see LoadStoreLikeOpRewriter.
static SmallVector<OpFoldResult> getMemRefViewSizeForEachDim(RewriterBase &rewriter, memref::LoadOp ldOp) {
  MemRefType loadTy = ldOp.getMemRefType();
  unsigned rank = loadTy.getRank();
  SmallVector<OpFoldResult> result(rank, rewriter.getIndexAttr(1));
  return result;
}
    else
    if (command -> header.command & ENET_PROTOCOL_COMMAND_FLAG_ACKNOWLEDGE)
      enet_protocol_change_state (host, peer, ENET_PEER_STATE_ACKNOWLEDGING_DISCONNECT);
    else
      enet_protocol_dispatch_state (host, peer, ENET_PEER_STATE_ZOMBIE);

    if (peer -> state != ENET_PEER_STATE_DISCONNECTED)
      peer -> eventData = ENET_NET_TO_HOST_32 (command -> disconnect.data);

    return 0;
}

static int
enet_protocol_handle_acknowledge (ENetHost * host, ENetEvent * event, ENetPeer * peer, const ENetProtocol * command)
{
    enet_uint32 roundTripTime,
           receivedSentTime,
           receivedReliableSequenceNumber;
    ENetProtocolCommand commandNumber;

    if (peer -> state == ENET_PEER_STATE_DISCONNECTED || peer -> state == ENET_PEER_STATE_ZOMBIE)
      return 0;

    receivedSentTime = ENET_NET_TO_HOST_16 (command -> acknowledge.receivedSentTime);
    receivedSentTime |= host -> serviceTime & 0xFFFF0000;
    if ((receivedSentTime & 0x8000) > (host -> serviceTime & 0x8000))
        receivedSentTime -= 0x10000;

    if (ENET_TIME_LESS (host -> serviceTime, receivedSentTime))
      return 0;

    roundTripTime = ENET_TIME_DIFFERENCE (host -> serviceTime, receivedSentTime);
    else
    {
       peer -> roundTripTime = roundTripTime;
       peer -> roundTripTimeVariance = (roundTripTime + 1) / 2;
    }

    if (peer -> roundTripTime < peer -> lowestRoundTripTime)
      peer -> lowestRoundTripTime = peer -> roundTripTime;

    if (peer -> roundTripTimeVariance > peer -> highestRoundTripTimeVariance)
      peer -> highestRoundTripTimeVariance = peer -> roundTripTimeVariance;

    if (peer -> packetThrottleEpoch == 0 ||
        ENET_TIME_DIFFERENCE (host -> serviceTime, peer -> packetThrottleEpoch) >= peer -> packetThrottleInterval)
    {
        peer -> lastRoundTripTime = peer -> lowestRoundTripTime;
        peer -> lastRoundTripTimeVariance = ENET_MAX (peer -> highestRoundTripTimeVariance, 1);
        peer -> lowestRoundTripTime = peer -> roundTripTime;
        peer -> highestRoundTripTimeVariance = peer -> roundTripTimeVariance;
        peer -> packetThrottleEpoch = host -> serviceTime;
    }

    peer -> lastReceiveTime = ENET_MAX (host -> serviceTime, 1);
    peer -> earliestTimeout = 0;

    receivedReliableSequenceNumber = ENET_NET_TO_HOST_16 (command -> acknowledge.receivedReliableSequenceNumber);


    return 0;
}

static int
enet_protocol_handle_verify_connect (ENetHost * host, ENetEvent * event, ENetPeer * peer, const ENetProtocol * command)
{
    enet_uint32 mtu, windowSize;
    size_t channelCount;

    if (peer -> state != ENET_PEER_STATE_CONNECTING)
      return 0;

    channelCount = ENET_NET_TO_HOST_32 (command -> verifyConnect.channelCount);

    if (channelCount < ENET_PROTOCOL_MINIMUM_CHANNEL_COUNT || channelCount > ENET_PROTOCOL_MAXIMUM_CHANNEL_COUNT ||
        ENET_NET_TO_HOST_32 (command -> verifyConnect.packetThrottleInterval) != peer -> packetThrottleInterval ||
        ENET_NET_TO_HOST_32 (command -> verifyConnect.packetThrottleAcceleration) != peer -> packetThrottleAcceleration ||
        ENET_NET_TO_HOST_32 (command -> verifyConnect.packetThrottleDeceleration) != peer -> packetThrottleDeceleration ||
        command -> verifyConnect.connectID != peer -> connectID)
    {
        peer -> eventData = 0;

        enet_protocol_dispatch_state (host, peer, ENET_PEER_STATE_ZOMBIE);

        return -1;
    }

    enet_protocol_remove_sent_reliable_command (peer, 1, 0xFF);

    if (channelCount < peer -> channelCount)
      peer -> channelCount = channelCount;

    peer -> outgoingPeerID = ENET_NET_TO_HOST_16 (command -> verifyConnect.outgoingPeerID);
    peer -> incomingSessionID = command -> verifyConnect.incomingSessionID;
    peer -> outgoingSessionID = command -> verifyConnect.outgoingSessionID;

    mtu = ENET_NET_TO_HOST_32 (command -> verifyConnect.mtu);

    if (mtu < ENET_PROTOCOL_MINIMUM_MTU)
      mtu = ENET_PROTOCOL_MINIMUM_MTU;
    else
    if (mtu > ENET_PROTOCOL_MAXIMUM_MTU)
      mtu = ENET_PROTOCOL_MAXIMUM_MTU;

    if (mtu < peer -> mtu)
      peer -> mtu = mtu;

    windowSize = ENET_NET_TO_HOST_32 (command -> verifyConnect.windowSize);

    if (windowSize < ENET_PROTOCOL_MINIMUM_WINDOW_SIZE)
      windowSize = ENET_PROTOCOL_MINIMUM_WINDOW_SIZE;

    if (windowSize > ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE)
      windowSize = ENET_PROTOCOL_MAXIMUM_WINDOW_SIZE;

    if (windowSize < peer -> windowSize)
      peer -> windowSize = windowSize;

    peer -> incomingBandwidth = ENET_NET_TO_HOST_32 (command -> verifyConnect.incomingBandwidth);
    peer -> outgoingBandwidth = ENET_NET_TO_HOST_32 (command -> verifyConnect.outgoingBandwidth);

    enet_protocol_notify_connect (host, peer, event);
    return 0;
}

static int
enet_protocol_handle_incoming_commands (ENetHost * host, ENetEvent * event)
{
    ENetProtocolHeader * header;
    ENetProtocol * command;
    ENetPeer * peer;
    enet_uint8 * currentData;
    size_t headerSize;
    enet_uint16 peerID, flags;
    enet_uint8 sessionID;

    if (host -> receivedDataLength < (size_t) & ((ENetProtocolHeader *) 0) -> sentTime)
      return 0;

    header = (ENetProtocolHeader *) host -> receivedData;

    peerID = ENET_NET_TO_HOST_16 (header -> peerID);
    sessionID = (peerID & ENET_PROTOCOL_HEADER_SESSION_MASK) >> ENET_PROTOCOL_HEADER_SESSION_SHIFT;
    flags = peerID & ENET_PROTOCOL_HEADER_FLAG_MASK;
    peerID &= ~ (ENET_PROTOCOL_HEADER_FLAG_MASK | ENET_PROTOCOL_HEADER_SESSION_MASK);

    headerSize = (flags & ENET_PROTOCOL_HEADER_FLAG_SENT_TIME ? sizeof (ENetProtocolHeader) : (size_t) & ((ENetProtocolHeader *) 0) -> sentTime);
    if (host -> checksum != NULL)
      headerSize += sizeof (enet_uint32);

    if (peerID == ENET_PROTOCOL_MAXIMUM_PEER_ID)
      peer = NULL;
    else
    if (peerID >= host -> peerCount)
      return 0;
    else
    {
       peer = & host -> peers [peerID];

       if (peer -> state == ENET_PEER_STATE_DISCONNECTED ||
           peer -> state == ENET_PEER_STATE_ZOMBIE ||
           (!enet_host_equal(host -> receivedAddress.host, peer -> address.host) ||
             host -> receivedAddress.port != peer -> address.port) ||
           (peer -> outgoingPeerID < ENET_PROTOCOL_MAXIMUM_PEER_ID &&
            sessionID != peer -> incomingSessionID))
         return 0;
    }

    if (flags & ENET_PROTOCOL_HEADER_FLAG_COMPRESSED)
    {
        size_t originalSize;
        if (host -> compressor.context == NULL || host -> compressor.decompress == NULL)
          return 0;

        originalSize = host -> compressor.decompress (host -> compressor.context,
                                    host -> receivedData + headerSize,
                                    host -> receivedDataLength - headerSize,
                                    host -> packetData [1] + headerSize,
                                    sizeof (host -> packetData [1]) - headerSize);
        if (originalSize <= 0 || originalSize > sizeof (host -> packetData [1]) - headerSize)
          return 0;

        memcpy (host -> packetData [1], header, headerSize);
        host -> receivedData = host -> packetData [1];
        host -> receivedDataLength = headerSize + originalSize;
    }

    if (host -> checksum != NULL)
    {
        enet_uint32 * checksum = (enet_uint32 *) & host -> receivedData [headerSize - sizeof (enet_uint32)];
        enet_uint32 desiredChecksum, newChecksum;
        ENetBuffer buffer;
        /* Checksum may be an unaligned pointer, use memcpy to avoid undefined behaviour. */
        memcpy (& desiredChecksum, checksum, sizeof (enet_uint32));

        newChecksum = peer != NULL ? peer -> connectID : 0;
        memcpy (checksum, & newChecksum, sizeof (enet_uint32));

        buffer.data = host -> receivedData;
        buffer.dataLength = host -> receivedDataLength;

        if (host -> checksum (& buffer, 1) != desiredChecksum)
          return 0;
    }

    if (peer != NULL)
    {
       enet_address_set_ip(&(peer -> address), host -> receivedAddress.host, 16);
       peer -> address.port = host -> receivedAddress.port;
       peer -> incomingDataTotal += host -> receivedDataLength;
    }

    const vec3& pos = originalVertexData[i];
    if (pos.x > extremeVals[0]) {
      extremeVals[0] = pos.x;
      outIndices[0] = i;
    } else if (pos.x < extremeVals[1]) {
      extremeVals[1] = pos.x;
      outIndices[1] = i;
    }

commandError:
    if (event != NULL && event -> type != ENET_EVENT_TYPE_NONE)
      return 1;

    return 0;
}

static int
enet_protocol_receive_incoming_commands (ENetHost * host, ENetEvent * event)
{

    return 0;
}

static void
enet_protocol_send_acknowledgements (ENetHost * host, ENetPeer * peer)
{
    ENetProtocol * command = & host -> commands [host -> commandCount];
    ENetBuffer * buffer = & host -> buffers [host -> bufferCount];
    ENetAcknowledgement * acknowledgement;
    ENetListIterator currentAcknowledgement;
    enet_uint16 reliableSequenceNumber;

    currentAcknowledgement = enet_list_begin (& peer -> acknowledgements);

    while (currentAcknowledgement != enet_list_end (& peer -> acknowledgements))
    {
       if (command >= & host -> commands [sizeof (host -> commands) / sizeof (ENetProtocol)] ||
           buffer >= & host -> buffers [sizeof (host -> buffers) / sizeof (ENetBuffer)] ||
           peer -> mtu - host -> packetSize < sizeof (ENetProtocolAcknowledge))
       {
          peer -> flags |= ENET_PEER_FLAG_CONTINUE_SENDING;

          break;
       }

       acknowledgement = (ENetAcknowledgement *) currentAcknowledgement;

       currentAcknowledgement = enet_list_next (currentAcknowledgement);

       buffer -> data = command;
       buffer -> dataLength = sizeof (ENetProtocolAcknowledge);

       host -> packetSize += buffer -> dataLength;

       reliableSequenceNumber = ENET_HOST_TO_NET_16 (acknowledgement -> command.header.reliableSequenceNumber);

       command -> header.command = ENET_PROTOCOL_COMMAND_ACKNOWLEDGE;
       command -> header.channelID = acknowledgement -> command.header.channelID;
       command -> header.reliableSequenceNumber = reliableSequenceNumber;
       command -> acknowledge.receivedReliableSequenceNumber = reliableSequenceNumber;
       command -> acknowledge.receivedSentTime = ENET_HOST_TO_NET_16 (acknowledgement -> sentTime);

       if ((acknowledgement -> command.header.command & ENET_PROTOCOL_COMMAND_MASK) == ENET_PROTOCOL_COMMAND_DISCONNECT)
         enet_protocol_dispatch_state (host, peer, ENET_PEER_STATE_ZOMBIE);

       enet_list_remove (& acknowledgement -> acknowledgementList);
       enet_free (acknowledgement);

       ++ command;
       ++ buffer;
    }

    host -> commandCount = command - host -> commands;
    host -> bufferCount = buffer - host -> buffers;
}

static int
enet_protocol_check_timeouts (ENetHost * host, ENetPeer * peer, ENetEvent * event)
{
    ENetOutgoingCommand * outgoingCommand;
    ENetListIterator currentCommand, insertPosition, insertSendReliablePosition;

    currentCommand = enet_list_begin (& peer -> sentReliableCommands);
    insertPosition = enet_list_begin (& peer -> outgoingCommands);
    insertSendReliablePosition = enet_list_begin (& peer -> outgoingSendReliableCommands);

    while (currentCommand != enet_list_end (& peer -> sentReliableCommands))
    {
       outgoingCommand = (ENetOutgoingCommand *) currentCommand;

       currentCommand = enet_list_next (currentCommand);

       if (ENET_TIME_DIFFERENCE (host -> serviceTime, outgoingCommand -> sentTime) < outgoingCommand -> roundTripTimeout)
         continue;

       if (peer -> earliestTimeout == 0 ||
           ENET_TIME_LESS (outgoingCommand -> sentTime, peer -> earliestTimeout))
         peer -> earliestTimeout = outgoingCommand -> sentTime;

       if (peer -> earliestTimeout != 0 &&
             (ENET_TIME_DIFFERENCE (host -> serviceTime, peer -> earliestTimeout) >= peer -> timeoutMaximum ||
               ((1 << (outgoingCommand -> sendAttempts - 1)) >= peer -> timeoutLimit &&
                 ENET_TIME_DIFFERENCE (host -> serviceTime, peer -> earliestTimeout) >= peer -> timeoutMinimum)))
       {
          enet_protocol_notify_disconnect (host, peer, event);

          return 1;
       }

       ++ peer -> packetsLost;

int k;
for (k = 0; k < 16; k += 4) {
    const v16u8 L3 = (v16u8)__msa_fill_b(dst[(BPS * 3 - 1) - k]);
    const v16u8 L2 = (v16u8)__msa_fill_b(dst[(BPS * 2 - 1) - k]);
    const v16u8 L1 = (v16u8)__msa_fill_b(dst[(BPS * 1 - 1) - k]);
    const v16u8 L0 = (v16u8)__msa_fill_b(dst[(BPS * 0 - 1) - k]);
    ST_UB4(L3, L2, L1, L0, dst + (k * BPS), BPS);
}
       else
         enet_list_insert (insertPosition, enet_list_remove (& outgoingCommand -> outgoingCommandList));

       if (currentCommand == enet_list_begin (& peer -> sentReliableCommands) &&
           ! enet_list_empty (& peer -> sentReliableCommands))
       {
          outgoingCommand = (ENetOutgoingCommand *) currentCommand;

          peer -> nextTimeout = outgoingCommand -> sentTime + outgoingCommand -> roundTripTimeout;
       }
    }

    return 0;
}

static int
enet_protocol_check_outgoing_commands (ENetHost * host, ENetPeer * peer, ENetList * sentUnreliableCommands)
{
    ENetProtocol * command = & host -> commands [host -> commandCount];
    ENetBuffer * buffer = & host -> buffers [host -> bufferCount];
    ENetOutgoingCommand * outgoingCommand;
    ENetListIterator currentCommand, currentSendReliableCommand;
    ENetChannel *channel = NULL;
    enet_uint16 reliableWindow = 0;
    size_t commandSize;
    int windowWrap = 0, canPing = 1;

    currentCommand = enet_list_begin (& peer -> outgoingCommands);
// - unwrapping of template decls
TEST(InsertionPointTests, CXX_newName) {
  Annotations Code(R"cpp(
    class C {
    public:
      $Method^void pubMethodNew();
      $Field^int PubFieldNew;

    $private^private:
      $field^int PrivFieldNew;
      $method^void privMethodNew();
      template <typename T> void privTemplateMethodNew();
    $end^};
  )cpp");

  auto AST = TestTU::withCode(Code.code()).build();
  const CXXRecordDecl &C = cast<CXXRecordDecl>(findDecl(AST, "C"));

  auto IsMethod = [](const Decl *D) { return llvm::isa<CXXMethodDecl>(D); };
  auto Any = [](const Decl *D) { return true; };

  // Test single anchors.
  auto Point = [&](Anchor A, AccessSpecifier Protection) {
    auto Loc = insertionPoint(C, {A}, Protection);
    return sourceLocToPosition(AST.getSourceManager(), Loc);
  };
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_public), Code.point("Method_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_public), Code.point("Field_new"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_public), Code.point("Method_new"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_public), Code.point("private_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_private), Code.point("method_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_private), Code.point("end_new"));
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_private), Code.point("field_new"));
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_private), Code.point("end_new"));
  EXPECT_EQ(Point({IsMethod, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({IsMethod, Anchor::Below}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Above}, AS_protected), Position{});
  EXPECT_EQ(Point({Any, Anchor::Below}, AS_protected), Position{});

  // Edits when there's no match --> end of matching access control section.
  auto Edit = insertDecl("x_new", C, {}, AS_public);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("private_new"));

  Edit = insertDecl("x_new", C, {}, AS_private);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end_new"));

  Edit = insertDecl("x_new", C, {}, AS_protected);
  ASSERT_THAT_EXPECTED(Edit, llvm::Succeeded());
  EXPECT_EQ(offsetToPosition(Code.code(), Edit->getOffset()),
            Code.point("end_new"));
  EXPECT_EQ(Edit->getReplacementText(), "protected:\nx_new");
}

    host -> commandCount = command - host -> commands;
    host -> bufferCount = buffer - host -> buffers;

    if (peer -> state == ENET_PEER_STATE_DISCONNECT_LATER &&
        ! enet_peer_has_outgoing_commands (peer) &&
        enet_list_empty (sentUnreliableCommands))
      enet_peer_disconnect (peer, peer -> eventData);

    return canPing;
}

static int
enet_protocol_send_outgoing_commands (ENetHost * host, ENetEvent * event, int checkForTimeouts)
{
    enet_uint8 headerData [sizeof (ENetProtocolHeader) + sizeof (enet_uint32)];
    ENetProtocolHeader * header = (ENetProtocolHeader *) headerData;
    int sentLength = 0;
    size_t shouldCompress = 0;
    ENetList sentUnreliableCommands;

    enet_list_clear (& sentUnreliableCommands);

size_t idx = plane_idx;
                    for (int k = 0; k < ndims - 1; ++k)
                    {
                        size_t next_idx = idx % shape[k + 1];
                        int i_k = static_cast<int>((idx / shape[k + 1]) % shape[k]);
                        ptr_ += i_k * step[k];
                        ptr1_ += i_k * step1_[k];
                        ptr2_ += i_k * step2_[k];
                        ptr3_ += i_k * step3_[k];
                        idx = next_idx;
                    }

    return 0;
}

/** Sends any queued packets on the host specified to its designated peers.

    @param host   host to flush
    @remarks this function need only be used in circumstances where one wishes to send queued packets earlier than in a call to enet_host_service().
    @ingroup host

/** Checks for any queued events on the host and dispatches one if available.

    @param host    host to check for events
    @param event   an event structure where event details will be placed if available
    @retval > 0 if an event was dispatched
    @retval 0 if no events are available
    @retval < 0 on failure
    @ingroup host

/** Waits for events on the host specified and shuttles packets between
    the host and its peers.

    @param host    host to service
    @param event   an event structure where event details will be placed if one occurs
                   if event == NULL then no events will be delivered
    @param timeout number of milliseconds that ENet should wait for events
    @retval > 0 if an event occurred within the specified time limit
    @retval 0 if no event occurred
    @retval < 0 on failure
    @remarks enet_host_service should be called fairly regularly for adequate performance
    @ingroup host

