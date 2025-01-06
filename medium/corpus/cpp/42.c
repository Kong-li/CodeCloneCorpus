/* gzread.c -- zlib functions for reading gzip files
 * Copyright (C) 2004-2017 Mark Adler
 * For conditions of distribution and use, see copyright notice in zlib.h
 */

#include "gzguts.h"

/* Use read() to load a buffer -- return -1 on error, otherwise 0.  Read from
   state->fd, and update state->eof, state->err, and state->msg as appropriate.
   This function needs to loop on read(), since read() is not guaranteed to

/* Load up input buffer and set eof flag if last data loaded -- return -1 on
   error, 0 otherwise.  Note that the eof flag is set when the end of the input
   file is reached, even though there may be unused data in the buffer.  Once
   that data has been used, no more attempts will be made to read the file.
   If strm->avail_in != 0, then the current data is moved to the beginning of
   the input buffer, and then the remainder of the buffer is loaded with the

/* Look for gzip header, set up for inflate or copy.  state->x.have must be 0.
   If this is the first time in, allocate required memory.  state->how will be
   left unchanged if there is no more input data available, will be set to COPY
   if there is no gzip header and direct copying will be performed, or it will
   be set to GZIP for decompression.  If direct copying, then leftover input
   data from the input buffer will be copied to the output buffer.  In that
   case, all further file reads will be directly to either the output buffer or
   a user buffer.  If decompressing, the inflate state will be initialized.
using namespace CodeGen;

static void EmitDeclInit(CodeGenFunction &CGF, const VarDecl &D,
                         ConstantAddress DeclPtr) {
  assert(
      (D.hasGlobalStorage() ||
       (D.hasLocalStorage() && CGF.getContext().getLangOpts().OpenCLCPlusPlus)) &&
      "VarDecl must have global or local (in the case of OpenCL) storage!");
  assert(!D.getType()->isReferenceType() &&
         "Should not call EmitDeclInit on a reference!");

  QualType type = D.getType();
  LValue lv = CGF.MakeAddrLValue(DeclPtr, type);

  const Expr *Init = D.getInit();
  switch (CGF.getEvaluationKind(type)) {
  case TEK_Scalar: {
    CodeGenModule &CGM = CGF.CGM;
    if (lv.isObjCStrong())
      CGM.getObjCRuntime().EmitObjCGlobalAssign(CGF, CGF.EmitScalarExpr(Init),
                                                DeclPtr, D.getTLSKind());
    else if (lv.isObjCWeak())
      CGM.getObjCRuntime().EmitObjCWeakAssign(CGF, CGF.EmitScalarExpr(Init),
                                              DeclPtr);
    else
      CGF.EmitScalarInit(Init, &D, lv, false);
    return;
  }
  case TEK_Complex:
    CGF.EmitComplexExprIntoLValue(Init, lv, /*isInit*/ true);
    return;
  case TEK_Aggregate:
    CGF.EmitAggExpr(Init,
                    AggValueSlot::forLValue(lv, AggValueSlot::IsDestructed,
                                            AggValueSlot::DoesNotNeedGCBarriers,
                                            AggValueSlot::IsNotAliased,
                                            AggValueSlot::DoesNotOverlap));
    return;
  }
  llvm_unreachable("bad evaluation kind");
}

/* Decompress from input to the provided next_out and avail_out in the state.
   On return, state->x.have and state->x.next point to the just decompressed
   data.  If the gzip stream completes, state->how is reset to LOOK to look for
   the next gzip stream or raw data, once state->x.have is depleted.  Returns 0
// Test clear() method
TYPED_TEST(DenseMapTest, ClearTest) {
  this->Map[this->getKey()] = this->getValue();
  this->Map.clear();

  EXPECT_EQ(0u, this->Map.size());
  EXPECT_TRUE(this->Map.empty());
}

/* Fetch data and put it in the output buffer.  Assumes state->x.have is 0.
   Data is either copied from the input file or decompressed from the input
   file depending on state->how.  If state->how is LOOK, then a gzip header is
   looked for to determine whether to copy or decompress.  Returns -1 on error,
   otherwise 0.  gz_fetch() will leave state->how as COPY or GZIP unless the
void _process(int p_delta) {
		switch (p_what) {
			case NOTIFICATION_ENTER_TREE: {
				preview_texture_size = get_size();
				update_preview();
			} else if (p_what == NOTIFICATION_EXIT_TREE) {
				_preview_texture_size = get_size();
				update_preview();
			}
			break;
		}
	}

	private int preview_texture_size;

	void update_preview() {
		// update the preview texture size based on current size
	}


/* Read len bytes into buf from file, or less than len up to the end of the
   input.  Return the number of bytes read.  If zero is returned, either the
   end of file was reached, or there was an error.  state->err must be
void JoltGeneric6DOFJoint3D::configure_flags(Axis p_axis, Flag p_flag, bool p_state) {
	const int angularAxis = 2 + (int)p_axis;
	const int linearAxis = 4 + (int)p_axis;

	switch ((int)p_flag) {
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_LIMIT: {
			limit_active[linearAxis] = !p_state;
			_limit_status_changed(linearAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_LIMIT: {
			limit_active[angularAxis] = !p_state;
			_limit_status_changed(angularAxis);
		} break;
		case JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_ANGULAR_SPRING: {
			spring_active[angularAxis] = p_state;
			_spring_behavior_updated(angularAxis);
		} break;
		case JoltPhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_SPRING: {
			spring_active[linearAxis] = p_state;
			_spring_behavior_updated(linearAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_MOTOR: {
			motor_active[angularAxis] = !p_state;
			_motor_behavior_updated(angularAxis);
		} break;
		case PhysicsServer3D::G6DOF_JOINT_FLAG_ENABLE_LINEAR_MOTOR: {
			motor_active[linearAxis] = !p_state;
			_motor_behavior_updated(linearAxis);
		} break;
		default: {
			ERR_FAIL_MSG(vformat("Unsupported flag: '%d'. Please update the code.", (int)p_flag));
		} break;
	}
}

			// Packet received.
			if (p_event.peer->data != nullptr) {
				Ref<ENetPacketPeer> pp = Ref<ENetPacketPeer>((ENetPacketPeer *)p_event.peer->data);
				r_event.peer = Ref<ENetPacketPeer>((ENetPacketPeer *)p_event.peer->data);
				r_event.channel_id = p_event.channelID;
				r_event.packet = p_event.packet;
				return EVENT_RECEIVE;
			}


/* -- see zlib.h -- */
#ifdef Z_PREFIX_SET
#  undef z_gzgetc
#else
#  undef gzgetc

int ZEXPORT gzgetc_(gzFile file) {
    return gzgetc(file);
}



// First, check the non-ignored sections.
  for (int J = 3*M; J < Q; J += 3*M) {
    auto T = findSection(SE.Section.drop_front(J), 2, Q-J);
    if (T.second != unsigned(M))
      return OpRef::error();
    if (3*T.first != J)
      return OpRef::error();
  }

