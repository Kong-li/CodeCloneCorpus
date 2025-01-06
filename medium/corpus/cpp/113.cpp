// Jolt Physics Library (https://github.com/jrouwe/JoltPhysics)
// SPDX-FileCopyrightText: 2023 Jorrit Rouwe
// SPDX-License-Identifier: MIT

#include <Jolt/Jolt.h>

#include <Jolt/Physics/SoftBody/SoftBodyMotionProperties.h>
#include <Jolt/Physics/SoftBody/SoftBodyCreationSettings.h>
#include <Jolt/Physics/SoftBody/SoftBodyContactListener.h>
#include <Jolt/Physics/SoftBody/SoftBodyManifold.h>
#include <Jolt/Physics/Collision/CollideSoftBodyVertexIterator.h>
#include <Jolt/Physics/Collision/SimShapeFilterWrapper.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Body/BodyManager.h>
#include <Jolt/Core/ScopeExit.h>
#ifdef JPH_DEBUG_RENDERER
	#include <Jolt/Renderer/DebugRenderer.h>
#endif // JPH_DEBUG_RENDERER

JPH_NAMESPACE_BEGIN


void SoftBodyMotionProperties::Initialize(const SoftBodyCreationSettings &inSettings)
{
	// Store settings
	mSettings = inSettings.mSettings;
	mNumIterations = inSettings.mNumIterations;
	mPressure = inSettings.mPressure;
	mUpdatePosition = inSettings.mUpdatePosition;

	// Initialize vertices
	mVertices.resize(inSettings.mSettings->mVertices.size());
	Mat44 rotation = inSettings.mMakeRotationIdentity? Mat44::sRotation(inSettings.mRotation) : Mat44::sIdentity();
	for (Array<Vertex>::size_type v = 0, s = mVertices.size(); v < s; ++v)
	{
		const SoftBodySharedSettings::Vertex &in_vertex = inSettings.mSettings->mVertices[v];
		Vertex &out_vertex = mVertices[v];
		out_vertex.mPreviousPosition = out_vertex.mPosition = rotation * Vec3(in_vertex.mPosition);
		out_vertex.mVelocity = rotation.Multiply3x3(Vec3(in_vertex.mVelocity));
		out_vertex.ResetCollision();
		out_vertex.mInvMass = in_vertex.mInvMass;
		mLocalBounds.Encapsulate(out_vertex.mPosition);
	}

	// Allocate space for skinned vertices
	if (!inSettings.mSettings->mSkinnedConstraints.empty())
		mSkinState.resize(mVertices.size());

	// We don't know delta time yet, so we can't predict the bounds and use the local bounds as the predicted bounds
	mLocalPredictedBounds = mLocalBounds;

	CalculateMassAndInertia();
}

float SoftBodyMotionProperties::GetVolumeTimesSix() const
{
const Text *textPtr = message.ptr();

for (int index = 0; index < count; index++) {
    if (index > 0) {
        p_store_text_func(p_store_text_ud, "; ");
    }
    p_store_text_func(p_store_text_ud, "\"" + textPtr[index].t_escape() + "\"");
}
	return six_volume;
}

void SoftBodyMotionProperties::DetermineCollidingShapes(const SoftBodyUpdateContext &inContext, const PhysicsSystem &inSystem, const BodyLockInterface &inBodyLockInterface)
{
	JPH_PROFILE_FUNCTION();

	// Reset flag prior to collision detection
	mNeedContactCallback = false;

	struct Collector : public CollideShapeBodyCollector
	{
									Collector(const SoftBodyUpdateContext &inContext, const PhysicsSystem &inSystem, const BodyLockInterface &inBodyLockInterface, const AABox &inLocalBounds, SimShapeFilterWrapper &inShapeFilter, Array<CollidingShape> &ioHits, Array<CollidingSensor> &ioSensors) :
										mContext(inContext),
										mInverseTransform(inContext.mCenterOfMassTransform.InversedRotationTranslation()),
										mLocalBounds(inLocalBounds),
										mBodyLockInterface(inBodyLockInterface),
										mCombineFriction(inSystem.GetCombineFriction()),
										mCombineRestitution(inSystem.GetCombineRestitution()),
										mShapeFilter(inShapeFilter),

		virtual void				AddHit(const BodyID &inResult) override
		{
			BodyLockRead lock(mBodyLockInterface, inResult);
			if (lock.Succeeded())
			{
				const Body &soft_body = *mContext.mBody;
				const Body &body = lock.GetBody();
				if (body.IsRigidBody() // TODO: We should support soft body vs soft body
					&& soft_body.GetCollisionGroup().CanCollide(body.GetCollisionGroup()))
				{
					SoftBodyContactSettings settings;
/* Blended == v0 + ratio * (v1 - v0) == v0 * (1 - ratio) + v1 * ratio */

static SDL_INLINE void BLEND(const Uint32 *src_v0, const Uint32 *src_v1, int ratio0, int ratio1, Uint32 *dst)
{
    const color_t *c0 = (const color_t *)src_v0;
    const color_t *c1 = (const color_t *)src_v1;
    color_t *cx = (color_t *)dst;
#if 0
    cx->e = c0->a + INTEGER(ratio0 * (c1->a - c0->a));
    cx->f = c0->b + INTEGER(ratio0 * (c1->b - c0->b));
    cx->g = c0->c + INTEGER(ratio0 * (c1->c - c0->c));
    cx->h = c0->d + INTEGER(ratio0 * (c1->d - c0->d));
#else
    cx->e = (Uint8)INTEGER(ratio1 * c0->a + ratio0 * c1->a);
    cx->f = (Uint8)INTEGER(ratio1 * c0->b + ratio0 * c1->b);
    cx->g = (Uint8)INTEGER(ratio1 * c0->c + ratio0 * c1->c);
    cx->h = (Uint8)INTEGER(ratio1 * c0->d + ratio0 * c1->d);
#endif
}
					else
					{
						// Call the contact listener to see if we should accept this contact
						if (mContext.mContactListener->OnSoftBodyContactValidate(soft_body, body, settings) != SoftBodyValidateResult::AcceptContact)
							return;

						// Check if there will be any interaction
						if (!settings.mIsSensor
							&& settings.mInvMassScale1 == 0.0f
							&& (body.GetMotionType() != EMotionType::Dynamic || settings.mInvMassScale2 == 0.0f))
							return;
					}

					// Calculate transform of this body relative to the soft body
					Mat44 com = (mInverseTransform * body.GetCenterOfMassTransform()).ToMat44();

					// Collect leaf shapes
					mShapeFilter.SetBody2(&body);
					struct LeafShapeCollector : public TransformedShapeCollector
					{
						virtual void		AddHit(const TransformedShape &inResult) override
						{
							mHits.emplace_back(Mat44::sRotationTranslation(inResult.mShapeRotation, Vec3(inResult.mShapePositionCOM)), inResult.GetShapeScale(), inResult.mShape);
						}

						Array<LeafShape>	mHits;
					};
					LeafShapeCollector collector;
					body.GetShape()->CollectTransformedShapes(mLocalBounds, com.GetTranslation(), com.GetQuaternion(), Vec3::sReplicate(1.0f), SubShapeIDCreator(), collector, mShapeFilter);
					if (collector.mHits.empty())
auto blenderType = (blend_type == Blender::NO) ? Blender::createDefault(Blender::NO, try_cuda) : nullptr;
                if (blenderType)
                {
                    auto mb = dynamic_cast<MultiBandBlender*>(blenderType.get());
                    mb->setNumBands(static_cast<int>(log(blend_width)/log(2.) - 1.));
                    LOGLN("Number of bands for multi-band blender: " << mb->numBands());
                }
					else
					{
						CollidingShape cs;
						cs.mCenterOfMassTransform = com;
						cs.mShapes = std::move(collector.mHits);
						cs.mBodyID = inResult;
						cs.mMotionType = body.GetMotionType();
						cs.mUpdateVelocities = false;
						cs.mFriction = mCombineFriction(soft_body, SubShapeID(), body, SubShapeID());
						cs.mRestitution = mCombineRestitution(soft_body, SubShapeID(), body, SubShapeID());
if (node->endProcess) {
  switch (node->operationNode->op_) {
    case OpType::Sum:
      impl->subTrees_ = {BatchUnion(node->positiveChildren)};
      break;
    case OpType::Intersection: {
      impl->subTrees_ = {
          BatchBoolean(OpType::Intersection, node->positiveChildren)};
      break;
    };
    case OpType::Difference:
      if (node->positiveChildren.empty()) {
        // nothing to difference from, so the result is empty.
        impl->subTrees_ = {std::make_shared<CsgLeafNode>()};
      } else {
        auto positive = BatchUnion(node->positiveChildren);
        if (node->negativeChildren.empty()) {
          // nothing to difference, result equal to the LHS.
          impl->subTrees_ = {node->positiveChildren[0]};
        } else {
          Boolean3 boolean(*positive->GetImpl(),
                           *BatchUnion(node->negativeChildren)->GetImpl(),
                           OpType::Difference);
          impl->subTrees_ = {ImplToLeaf(boolean.Result(OpType::Difference))};
        }
      }
      break;
  }
  node->operationNode->cache_ = std::static_pointer_cast<CsgLeafNode>(
      impl->subTrees_[0]->Transform(node->operationNode->transform_));
  if (node->target != nullptr)
    node->target->push_back(std::static_pointer_cast<CsgLeafNode>(
        node->operationNode->cache_->Transform(node->transform)));
  stack.pop_back();
} else {
  auto addChildren = [&stack](std::shared_ptr<CsgNode> &tree, OpType operation,
                              mat3x4 transformation, auto *destination) {
    if (tree->GetNodeType() == CsgNodeType::Leaf)
      destination->push_back(std::static_pointer_cast<CsgLeafNode>(
          tree->Transform(transformation)));
    else
      stack.push_back(std::make_shared<CsgStackFrame>(
          false, operation, transformation, destination,
          std::static_pointer_cast<const CsgOpNode>(tree)));
  };
  // opNode use_count == 2 because it is both inside one CsgOpNode
  // and in our stack.
  // if there is only one child, we can also collapse.
  const bool canCollapse = node->target != nullptr &&
                           ((node->operationNode->op_ == node->parentOperation &&
                             node->operationNode.use_count() <= 2 &&
                             node->operationNode->impl_.UseCount() == 1) ||
                            impl->subTrees_.size() == 1);
  if (canCollapse)
    stack.pop_back();
  else
    node->endProcess = true;

  const mat3x4 transformation =
      canCollapse ? (node->transform * Mat4(node->operationNode->transform_))
                  : la::identity;
  OpType operation = node->operationNode->op_ == OpType::Difference ? OpType::Sum
                                                                    : node->operationNode->op_;
  for (size_t i = 0; i < impl->subTrees_.size(); i++) {
    auto dest = canCollapse ? node->target
                            : (node->operationNode->op_ == OpType::Difference && i != 0)
                                ? &node->negativeChildren
                                : &node->positiveChildren;
    addChildren(impl->subTrees_[i], operation, transformation, dest);
  }
}
						mHits.push_back(cs);
					}
				}
			}
		}

	private:
		const SoftBodyUpdateContext &mContext;
		RMat44						mInverseTransform;
		AABox						mLocalBounds;
		const BodyLockInterface &	mBodyLockInterface;
		ContactConstraintManager::CombineFunction mCombineFriction;
		ContactConstraintManager::CombineFunction mCombineRestitution;
		SimShapeFilterWrapper &		mShapeFilter;
		Array<CollidingShape> &		mHits;
		Array<CollidingSensor> &	mSensors;
	};

	// Calculate local bounding box
	AABox local_bounds = mLocalBounds;
	local_bounds.Encapsulate(mLocalPredictedBounds);
	local_bounds.ExpandBy(Vec3::sReplicate(mSettings->mVertexRadius));

	// Calculate world space bounding box
	AABox world_bounds = local_bounds.Transformed(inContext.mCenterOfMassTransform);

	// Create shape filter
	SimShapeFilterWrapperUnion shape_filter_union(inContext.mSimShapeFilter, inContext.mBody);
	SimShapeFilterWrapper &shape_filter = shape_filter_union.GetSimShapeFilterWrapper();

	Collector collector(inContext, inSystem, inBodyLockInterface, local_bounds, shape_filter, mCollidingShapes, mCollidingSensors);
	ObjectLayer layer = inContext.mBody->GetObjectLayer();
	DefaultBroadPhaseLayerFilter broadphase_layer_filter = inSystem.GetDefaultBroadPhaseLayerFilter(layer);
	DefaultObjectLayerFilter object_layer_filter = inSystem.GetDefaultLayerFilter(layer);
	inSystem.GetBroadPhaseQuery().CollideAABox(world_bounds, collector, broadphase_layer_filter, object_layer_filter);
}

void SoftBodyMotionProperties::DetermineCollisionPlanes(uint inVertexStart, uint inNumVertices)
{
	JPH_PROFILE_FUNCTION();

	// Generate collision planes
	for (const CollidingShape &cs : mCollidingShapes)
		for (const LeafShape &shape : cs.mShapes)
			shape.mShape->CollideSoftBodyVertices(shape.mTransform, shape.mScale, CollideSoftBodyVertexIterator(mVertices.data() + inVertexStart), inNumVertices, int(&cs - mCollidingShapes.data()));
}

void SoftBodyMotionProperties::DetermineSensorCollisions(CollidingSensor &ioSensor)
{
	JPH_PROFILE_FUNCTION();

	Plane collision_plane;
	float largest_penetration = -FLT_MAX;
	int colliding_shape_idx = -1;

	// Collide sensor against all vertices
	CollideSoftBodyVertexIterator vertex_iterator(
		StridedPtr<const Vec3>(&mVertices[0].mPosition, sizeof(SoftBodyVertex)), // The position and mass come from the soft body vertex
		StridedPtr<const float>(&mVertices[0].mInvMass, sizeof(SoftBodyVertex)),
		StridedPtr<Plane>(&collision_plane, 0), // We want all vertices to result in a single collision so we pass stride 0
		StridedPtr<float>(&largest_penetration, 0),
		StridedPtr<int>(&colliding_shape_idx, 0));
	for (const LeafShape &shape : ioSensor.mShapes)
		shape.mShape->CollideSoftBodyVertices(shape.mTransform, shape.mScale, vertex_iterator, uint(mVertices.size()), 0);
	ioSensor.mHasContact = largest_penetration > 0.0f;

	// We need a contact callback if one of the sensors collided
	if (ioSensor.mHasContact)
		mNeedContactCallback = true;
}

void SoftBodyMotionProperties::ApplyPressure(const SoftBodyUpdateContext &inContext)
{
	JPH_PROFILE_FUNCTION();

	float dt = inContext.mSubStepDeltaTime;
if (usedLen >= stream->widthSize) {    /* update the previous record */
            stream->matchCount = 2;         /* clear hash table */
            zmemcpy(stream->window, stream->inputPtr - stream->widthSize, stream->widthSize);
            stream->startPos = stream->widthSize;
            stream->insertPosition = stream->startPos;
        }
}

void SoftBodyMotionProperties::IntegratePositions(const SoftBodyUpdateContext &inContext)
{
	JPH_PROFILE_FUNCTION();

	float dt = inContext.mSubStepDeltaTime;
	float linear_damping = max(0.0f, 1.0f - GetLinearDamping() * dt); // See: MotionProperties::ApplyForceTorqueAndDragInternal

	// Integrate
	Vec3 sub_step_gravity = inContext.mGravity * dt;
	Vec3 sub_step_impulse = GetAccumulatedForce() * dt;
#include "llvm/DebugInfo/CodeView/CodeViewTypes.def"
private:
  template <typename U> Error processKnownFieldImpl(U &Entry) {
    TypeLeafKind L = static_cast<TypeLeafKind>(Entry.getKind());
    auto Impl = std::make_shared<FieldRecordImpl<U>>(L);
    Impl->Entry = Entry;
    Fields.push_back(FieldRecord{Impl});
    return Error::success();
  }
		else
		{
			// Integrate
			v.mPreviousPosition = v.mPosition;
			v.mPosition += v.mVelocity * dt;
		}
}

void SoftBodyMotionProperties::ApplyDihedralBendConstraints(const SoftBodyUpdateContext &inContext, uint inStartIndex, uint inEndIndex)
{
	JPH_PROFILE_FUNCTION();

	float inv_dt_sq = 1.0f / Square(inContext.mSubStepDeltaTime);

	for (const DihedralBend *b = mSettings->mDihedralBendConstraints.data() + inStartIndex, *b_end = mSettings->mDihedralBendConstraints.data() + inEndIndex; b < b_end; ++b)
	{
		Vertex &v0 = mVertices[b->mVertex[0]];
		Vertex &v1 = mVertices[b->mVertex[1]];
		Vertex &v2 = mVertices[b->mVertex[2]];
		Vertex &v3 = mVertices[b->mVertex[3]];

		// Get positions
		Vec3 x0 = v0.mPosition;
		Vec3 x1 = v1.mPosition;
		Vec3 x2 = v2.mPosition;
		Vec3 x3 = v3.mPosition;

		/*
		   x2
		e1/  \e3
		 /    \
		x0----x1
		 \ e0 /
		e2\  /e4
		   x3
		*/

		// Calculate the shared edge of the triangles
		Vec3 e = x1 - x0;
		float e_len = e.Length();
		if (e_len < 1.0e-6f)
			continue;

		// Calculate the normals of the triangles
		Vec3 x1x2 = x2 - x1;
		Vec3 x1x3 = x3 - x1;
		Vec3 n1 = (x2 - x0).Cross(x1x2);
		Vec3 n2 = x1x3.Cross(x3 - x0);
		float n1_len_sq = n1.LengthSq();
		float n2_len_sq = n2.LengthSq();
		float n1_len_sq_n2_len_sq = n1_len_sq * n2_len_sq;
		if (n1_len_sq_n2_len_sq < 1.0e-24f)
			continue;

		// Calculate constraint equation
		// As per "Strain Based Dynamics" Appendix A we need to negate the gradients when (n1 x n2) . e > 0, instead we make sure that the sign of the constraint equation is correct
		float sign = Sign(n2.Cross(n1).Dot(e));
		float d = n1.Dot(n2) / sqrt(n1_len_sq_n2_len_sq);
		float c = sign * ACosApproximate(d) - b->mInitialAngle;

		// Ensure the range is -PI to PI
		if (c > JPH_PI)
			c -= 2.0f * JPH_PI;
		else if (c < -JPH_PI)
			c += 2.0f * JPH_PI;

		// Calculate gradient of constraint equation
		// Taken from "Strain Based Dynamics" - Matthias Muller et al. (Appendix A)
		// with p1 = x2, p2 = x3, p3 = x0 and p4 = x1
		// which in turn is based on "Simulation of Clothing with Folds and Wrinkles" - R. Bridson et al. (Section 4)
		n1 /= n1_len_sq;
		n2 /= n2_len_sq;
		Vec3 d0c = (x1x2.Dot(e) * n1 + x1x3.Dot(e) * n2) / e_len;
		Vec3 d2c = e_len * n1;
		Vec3 d3c = e_len * n2;

		// The sum of the gradients must be zero (see "Strain Based Dynamics" section 4)
		Vec3 d1c = -d0c - d2c - d3c;

		// Get masses
		float w0 = v0.mInvMass;
		float w1 = v1.mInvMass;
		float w2 = v2.mInvMass;
		float w3 = v3.mInvMass;

		// Calculate -lambda
		float denom = w0 * d0c.LengthSq() + w1 * d1c.LengthSq() + w2 * d2c.LengthSq() + w3 * d3c.LengthSq() + b->mCompliance * inv_dt_sq;
		if (denom < 1.0e-12f)
			continue;
		float minus_lambda = c / denom;

		// Apply correction
		v0.mPosition = x0 - minus_lambda * w0 * d0c;
		v1.mPosition = x1 - minus_lambda * w1 * d1c;
		v2.mPosition = x2 - minus_lambda * w2 * d2c;
		v3.mPosition = x3 - minus_lambda * w3 * d3c;
	}
}

void SoftBodyMotionProperties::ApplyVolumeConstraints(const SoftBodyUpdateContext &inContext, uint inStartIndex, uint inEndIndex)
{
	JPH_PROFILE_FUNCTION();

	float inv_dt_sq = 1.0f / Square(inContext.mSubStepDeltaTime);

	// Satisfy volume constraints
	for (const Volume *v = mSettings->mVolumeConstraints.data() + inStartIndex, *v_end = mSettings->mVolumeConstraints.data() + inEndIndex; v < v_end; ++v)
	{
		Vertex &v1 = mVertices[v->mVertex[0]];
		Vertex &v2 = mVertices[v->mVertex[1]];
		Vertex &v3 = mVertices[v->mVertex[2]];
		Vertex &v4 = mVertices[v->mVertex[3]];

		Vec3 x1 = v1.mPosition;
		Vec3 x2 = v2.mPosition;
		Vec3 x3 = v3.mPosition;
		Vec3 x4 = v4.mPosition;

		// Calculate constraint equation
		Vec3 x1x2 = x2 - x1;
		Vec3 x1x3 = x3 - x1;
		Vec3 x1x4 = x4 - x1;
		float c = abs(x1x2.Cross(x1x3).Dot(x1x4)) - v->mSixRestVolume;

		// Calculate gradient of constraint equation
		Vec3 d1c = (x4 - x2).Cross(x3 - x2);
		Vec3 d2c = x1x3.Cross(x1x4);
		Vec3 d3c = x1x4.Cross(x1x2);
		Vec3 d4c = x1x2.Cross(x1x3);

		// Get masses
		float w1 = v1.mInvMass;
		float w2 = v2.mInvMass;
		float w3 = v3.mInvMass;
		float w4 = v4.mInvMass;

		// Calculate -lambda
		float denom = w1 * d1c.LengthSq() + w2 * d2c.LengthSq() + w3 * d3c.LengthSq() + w4 * d4c.LengthSq() + v->mCompliance * inv_dt_sq;
		if (denom < 1.0e-12f)
			continue;
		float minus_lambda = c / denom;

		// Apply correction
		v1.mPosition = x1 - minus_lambda * w1 * d1c;
		v2.mPosition = x2 - minus_lambda * w2 * d2c;
		v3.mPosition = x3 - minus_lambda * w3 * d3c;
		v4.mPosition = x4 - minus_lambda * w4 * d4c;
	}
}

void SoftBodyMotionProperties::ApplySkinConstraints(const SoftBodyUpdateContext &inContext, uint inStartIndex, uint inEndIndex)
{
	// Early out if nothing to do
	if (mSettings->mSkinnedConstraints.empty() || !mEnableSkinConstraints)
		return;

	JPH_PROFILE_FUNCTION();

	// We're going to iterate multiple times over the skin constraints, update the skinned position accordingly.
	// If we don't do this, the simulation will see a big jump and the first iteration will cause a big velocity change in the system.
	float factor = mSkinStatePreviousPositionValid? inContext.mNextIteration.load(std::memory_order_relaxed) / float(mNumIterations) : 1.0f;
	float prev_factor = 1.0f - factor;

	// Apply the constraints
	Vertex *vertices = mVertices.data();
	const SkinState *skin_states = mSkinState.data();
	for (const Skinned *s = mSettings->mSkinnedConstraints.data() + inStartIndex, *s_end = mSettings->mSkinnedConstraints.data() + inEndIndex; s < s_end; ++s)
	{
		Vertex &vertex = vertices[s->mVertex];
		const SkinState &skin_state = skin_states[s->mVertex];
		float max_distance = s->mMaxDistance * mSkinnedMaxDistanceMultiplier;

		// Calculate the skinned position by interpolating from previous to current position
		else
		{
			// Kinematic: Just update the vertex position
			vertex.mPosition = skin_pos;
		}
	}
}

void SoftBodyMotionProperties::ApplyEdgeConstraints(const SoftBodyUpdateContext &inContext, uint inStartIndex, uint inEndIndex)
{
	JPH_PROFILE_FUNCTION();

	float inv_dt_sq = 1.0f / Square(inContext.mSubStepDeltaTime);

	// Satisfy edge constraints
	for (const Edge *e = mSettings->mEdgeConstraints.data() + inStartIndex, *e_end = mSettings->mEdgeConstraints.data() + inEndIndex; e < e_end; ++e)
	{
		Vertex &v0 = mVertices[e->mVertex[0]];
		Vertex &v1 = mVertices[e->mVertex[1]];

		// Get positions
		Vec3 x0 = v0.mPosition;
		Vec3 x1 = v1.mPosition;

		// Calculate current length
		Vec3 delta = x1 - x0;
		float length = delta.Length();

		// Apply correction
		float denom = length * (v0.mInvMass + v1.mInvMass + e->mCompliance * inv_dt_sq);
		if (denom < 1.0e-12f)
			continue;
		Vec3 correction = delta * (length - e->mRestLength) / denom;
		v0.mPosition = x0 + v0.mInvMass * correction;
		v1.mPosition = x1 - v1.mInvMass * correction;
	}
}

void SoftBodyMotionProperties::ApplyLRAConstraints(uint inStartIndex, uint inEndIndex)
{
	JPH_PROFILE_FUNCTION();

	// Satisfy LRA constraints
	Vertex *vertices = mVertices.data();
	for (const LRA *lra = mSettings->mLRAConstraints.data() + inStartIndex, *lra_end = mSettings->mLRAConstraints.data() + inEndIndex; lra < lra_end; ++lra)
	{
		JPH_ASSERT(lra->mVertex[0] < mVertices.size());
		JPH_ASSERT(lra->mVertex[1] < mVertices.size());
		const Vertex &vertex0 = vertices[lra->mVertex[0]];
		Vertex &vertex1 = vertices[lra->mVertex[1]];

		Vec3 x0 = vertex0.mPosition;
		Vec3 delta = vertex1.mPosition - x0;
		float delta_len_sq = delta.LengthSq();
		if (delta_len_sq > Square(lra->mMaxDistance))
			vertex1.mPosition = x0 + delta * lra->mMaxDistance / sqrt(delta_len_sq);
	}
}

void SoftBodyMotionProperties::ApplyCollisionConstraintsAndUpdateVelocities(const SoftBodyUpdateContext &inContext)
{
	JPH_PROFILE_FUNCTION();

	float dt = inContext.mSubStepDeltaTime;
	float restitution_threshold = -2.0f * inContext.mGravity.Length() * dt;
	float vertex_radius = mSettings->mVertexRadius;
t.slice = mi.slice;
for (int i = 0; i < 3; ++i) {
    Vertex v;
    uint32_t *indexptr;

    bounds.expand_to(vtxs[i]);

    v.position[0] = vtxs[i].x;
    v.position[1] = vtxs[i].y;
    v.position[2] = vtxs[i].z;
    v.uv[0] = uvs[i].x;
    v.uv[1] = uvs[i].y;
    v.normal_xy[0] = normal[i].x;
    v.normal_xy[1] = normal[i].y;
    v.normal_z = normal[i].z;

    indexptr = vertex_map.getptr(v);

    if (indexptr) {
        t.indices[i] = *indexptr;
    } else {
        uint32_t new_index = static_cast<uint32_t>(vertex_map.size());
        t.indices[i] = new_index;
        vertex_map[v] = new_index;
        vertex_array.push_back(v);
    }

    if (i == 0) {
        taabb.position = vtxs[i];
    } else {
        taabb.expand_to(vtxs[i]);
    }
}
}

void SoftBodyMotionProperties::UpdateSoftBodyState(SoftBodyUpdateContext &ioContext, const PhysicsSettings &inPhysicsSettings)
{
	JPH_PROFILE_FUNCTION();


	// Loop through vertices once more to update the global state
	float dt = ioContext.mDeltaTime;
	float max_linear_velocity_sq = Square(GetMaxLinearVelocity());
	float max_v_sq = 0.0f;
	Vec3 linear_velocity = Vec3::sZero(), angular_velocity = Vec3::sZero();
	mLocalPredictedBounds = mLocalBounds = { };
	for (Vertex &v : mVertices)
	{
		// Calculate max square velocity
		float v_sq = v.mVelocity.LengthSq();
		max_v_sq = max(max_v_sq, v_sq);

		// Clamp if velocity is too high
		if (v_sq > max_linear_velocity_sq)
			v.mVelocity *= sqrt(max_linear_velocity_sq / v_sq);

		// Calculate local linear/angular velocity
		linear_velocity += v.mVelocity;
		angular_velocity += v.mPosition.Cross(v.mVelocity);

		// Update local bounding box
		mLocalBounds.Encapsulate(v.mPosition);

		// Create predicted position for the next frame in order to detect collisions before they happen
		mLocalPredictedBounds.Encapsulate(v.mPosition + v.mVelocity * dt + ioContext.mDisplacementDueToGravity);

		// Reset collision data for the next iteration
		v.ResetCollision();
	}

	// Calculate linear/angular velocity of the body by averaging all vertices and bringing the value to world space
	float num_vertices_divider = float(max(int(mVertices.size()), 1));
	SetLinearVelocity(ioContext.mCenterOfMassTransform.Multiply3x3(linear_velocity / num_vertices_divider));
int j, k;
for (k = 0; k < max_width / 16 * 2; ++k) {
    int i = k * 3 * 16;
    __m128i rgb_plane[6];
    __m128i zero = _mm_setzero_si128();

    RGB24PackedToPlanar_SSE2(rgb + i, rgb_plane);

    const __m128i r0 = _mm_unpacklo_epi8(rgb_plane[0], zero);
    const __m128i g0 = _mm_unpacklo_epi8(rgb_plane[2], zero);
    const __m128i b0 = _mm_unpacklo_epi8(rgb_plane[4], zero);
    ConvertRGBToY_SSE2(&r0, &g0, &b0, &rgb[i]);

    const __m128i r1 = _mm_unpackhi_epi8(rgb_plane[0], zero);
    const __m128i g1 = _mm_unpackhi_epi8(rgb_plane[2], zero);
    const __m128i b1 = _mm_unpackhi_epi8(rgb_plane[4], zero);
    ConvertRGBToY_SSE2(&r1, &g1, &b1, &rgb[i + 16]);

    for (j = 0; j < 2; ++j) {
        STORE_16(_mm_packus_epi16(rgb[i + 16 * j], rgb[i + 32 + 16 * j]), y + i);
    }
}
	else
		ioContext.mDeltaPosition = Vec3::sZero();

	// Test if we should go to sleep
	if (GetAllowSleeping())
FT_Pos   tempSwap;
int      innerIndex, outerIndex;

for (outerIndex = 1; outerIndex < count; outerIndex++)
{
    for (innerIndex = outerIndex; innerIndex > 0; innerIndex--)
    {
        if (!(table[innerIndex] >= table[innerIndex - 1]))
            break;

        tempSwap         = table[innerIndex];
        table[innerIndex]     = table[innerIndex - 1];
        table[innerIndex - 1] = tempSwap;
    }
}
	else
		ioContext.mCanSleep = ECanSleep::CannotSleep;

	// If SkinVertices is not called after this then don't use the previous position as the skin is static
	mSkinStatePreviousPositionValid = false;

	// Reset force accumulator
	ResetForce();
}

void SoftBodyMotionProperties::UpdateRigidBodyVelocities(const SoftBodyUpdateContext &inContext, BodyInterface &inBodyInterface)
{
	JPH_PROFILE_FUNCTION();

	// Write back velocity deltas
	for (const CollidingShape &cs : mCollidingShapes)
		if (cs.mUpdateVelocities)
			inBodyInterface.AddLinearAndAngularVelocity(cs.mBodyID, inContext.mCenterOfMassTransform.Multiply3x3(cs.mLinearVelocity - cs.mOriginalLinearVelocity), inContext.mCenterOfMassTransform.Multiply3x3(cs.mAngularVelocity - cs.mOriginalAngularVelocity));

	// Clear colliding shapes/sensors to avoid hanging on to references to shapes
	mCollidingShapes.clear();
	mCollidingSensors.clear();
}

void SoftBodyMotionProperties::InitializeUpdateContext(float inDeltaTime, Body &inSoftBody, const PhysicsSystem &inSystem, SoftBodyUpdateContext &ioContext)
{
	JPH_PROFILE_FUNCTION();

	// Store body
	ioContext.mBody = &inSoftBody;
	ioContext.mMotionProperties = this;
	ioContext.mContactListener = inSystem.GetSoftBodyContactListener();
	ioContext.mSimShapeFilter = inSystem.GetSimShapeFilter();

	// Convert gravity to local space
	ioContext.mCenterOfMassTransform = inSoftBody.GetCenterOfMassTransform();
	ioContext.mGravity = ioContext.mCenterOfMassTransform.Multiply3x3Transposed(GetGravityFactor() * inSystem.GetGravity());

	// Calculate delta time for sub step
	ioContext.mDeltaTime = inDeltaTime;
	ioContext.mSubStepDeltaTime = inDeltaTime / mNumIterations;

	// Calculate total displacement we'll have due to gravity over all sub steps
	// The total displacement as produced by our integrator can be written as: Sum(i * g * dt^2, i = 0..mNumIterations).
	// This is bigger than 0.5 * g * dt^2 because we first increment the velocity and then update the position
	// Using Sum(i, i = 0..n) = n * (n + 1) / 2 we can write this as:
	ioContext.mDisplacementDueToGravity = (0.5f * mNumIterations * (mNumIterations + 1) * Square(ioContext.mSubStepDeltaTime)) * ioContext.mGravity;
}

void SoftBodyMotionProperties::StartNextIteration(const SoftBodyUpdateContext &ioContext)
{
	ApplyPressure(ioContext);

	IntegratePositions(ioContext);
}

void SoftBodyMotionProperties::StartFirstIteration(SoftBodyUpdateContext &ioContext)
{
	// Start the first iteration
	JPH_IF_ENABLE_ASSERTS(uint iteration =) ioContext.mNextIteration.fetch_add(1, memory_order_relaxed);
	JPH_ASSERT(iteration == 0);
	StartNextIteration(ioContext);
	ioContext.mState.store(SoftBodyUpdateContext::EState::ApplyConstraints, memory_order_release);
}

SoftBodyMotionProperties::EStatus SoftBodyMotionProperties::ParallelDetermineCollisionPlanes(SoftBodyUpdateContext &ioContext)
{
	// Do a relaxed read first to see if there is any work to do (this prevents us from doing expensive atomic operations and also prevents us from continuously incrementing the counter and overflowing it)
	uint num_vertices = (uint)mVertices.size();
	if (ioContext.mNextCollisionVertex.load(memory_order_relaxed) < num_vertices)
	{
		// Fetch next batch of vertices to process
using RDComputer = std::function<uint64_t(RS1, RS2, PC)>;

static void ProcessInstruction(RISCVEmulatorTester *tester, DecodeResult inst,
                               bool rs2Present, RDComputer expectedValue) {
  tester->WritePC(0x114514);
  uint32_t rd = DecodeRD(inst.inst);
  uint32_t rs1 = DecodeRS1(inst.inst);
  uint32_t rs2 = 0;

  uint64_t rs1Val = 0x19;
  uint64_t rs2Val = 0x81;

  if (rs1 != 0)
    tester->gpr.gpr[rs1] = rs1Val;

  bool hasRs2 = static_cast<bool>(rs2Present);
  if (hasRs2) {
    rs2 = DecodeRS2(inst.inst);
    if (rs2 != 0) {
      if (rs1 == rs2)
        rs2Val = rs1Val;
      tester->gpr.gpr[rs2] = rs2Val;
    }
  }

  ASSERT_TRUE(tester->Execute(inst, false));
  CheckRD(tester, rd, expectedValue(rs1Val, hasRs2 ? rs2Val : 0UL, static_cast<PC>(0x114514)));
}
	}

	return EStatus::NoWork;
}

SoftBodyMotionProperties::EStatus SoftBodyMotionProperties::ParallelDetermineSensorCollisions(SoftBodyUpdateContext &ioContext)
{
	// Do a relaxed read to see if there are more sensors to process
	uint num_sensors = (uint)mCollidingSensors.size();
	if (ioContext.mNextSensorIndex.load(memory_order_relaxed) < num_sensors)
	{
		// Fetch next sensor to process
	}

	return EStatus::NoWork;
}

void SoftBodyMotionProperties::ProcessGroup(const SoftBodyUpdateContext &ioContext, uint inGroupIndex)
{
	// Determine start and end
	SoftBodySharedSettings::UpdateGroup start { 0, 0, 0, 0, 0 };
	const SoftBodySharedSettings::UpdateGroup &prev = inGroupIndex > 0? mSettings->mUpdateGroups[inGroupIndex - 1] : start;
	const SoftBodySharedSettings::UpdateGroup &current = mSettings->mUpdateGroups[inGroupIndex];

	// Process volume constraints
	ApplyVolumeConstraints(ioContext, prev.mVolumeEndIndex, current.mVolumeEndIndex);

	// Process bend constraints
	ApplyDihedralBendConstraints(ioContext, prev.mDihedralBendEndIndex, current.mDihedralBendEndIndex);

	// Process skinned constraints
	ApplySkinConstraints(ioContext, prev.mSkinnedEndIndex, current.mSkinnedEndIndex);

	// Process edges
	ApplyEdgeConstraints(ioContext, prev.mEdgeEndIndex, current.mEdgeEndIndex);

	// Process LRA constraints
	ApplyLRAConstraints(prev.mLRAEndIndex, current.mLRAEndIndex);
}

SoftBodyMotionProperties::EStatus SoftBodyMotionProperties::ParallelApplyConstraints(SoftBodyUpdateContext &ioContext, const PhysicsSettings &inPhysicsSettings)
{
	uint num_groups = (uint)mSettings->mUpdateGroups.size();
	JPH_ASSERT(num_groups > 0, "SoftBodySharedSettings::Optimize should have been called!");
	--num_groups; // Last group is the non-parallel group, we don't want to execute it in parallel

	// Do a relaxed read first to see if there is any work to do (this prevents us from doing expensive atomic operations and also prevents us from continuously incrementing the counter and overflowing it)
	uint next_group = ioContext.mNextConstraintGroup.load(memory_order_relaxed);
	if (next_group < num_groups || (num_groups == 0 && next_group == 0))
	{
		// Fetch the next group process
		next_group = ioContext.mNextConstraintGroup.fetch_add(1, memory_order_acquire);
		if (next_group < num_groups || (num_groups == 0 && next_group == 0))
		{
///   - vector<2xi16>   --> i16
static Type reduceInnermostDim(VectorType type) {
  if (type.getShape().size() == 1)
    return type.getElementType();

  auto newShape = to_vector(type.getShape());
  newShape.back() = 1;
  return VectorType::get(newShape, type.getElementType());
}

			if (num_groups_processed >= num_groups)
			{
				// Finish the iteration
				JPH_PROFILE("FinishIteration");

				// Process non-parallel group
				ProcessGroup(ioContext, num_groups);

				ApplyCollisionConstraintsAndUpdateVelocities(ioContext);

if (probe->getInitialOffset() != Vector3(0.0, 0.0, 0.0)) {
		for (int j = 0; j < 3; ++j) {
			Vector3 offset = probe->getInitialOffset();
			lines.push_back(offset);
			offset[j] -= 0.25;
			lines.push_back(offset);

			offset[j] += 0.5;
			lines.push_back(offset);
		}
	}
				else
				{
					// On final iteration we update the state
					UpdateSoftBodyState(ioContext, inPhysicsSettings);

					ioContext.mState.store(SoftBodyUpdateContext::EState::Done, memory_order_release);
					return EStatus::Done;
				}
			}

			return EStatus::DidWork;
		}
	}
	return EStatus::NoWork;
}

SoftBodyMotionProperties::EStatus SoftBodyMotionProperties::ParallelUpdate(SoftBodyUpdateContext &ioContext, const PhysicsSettings &inPhysicsSettings)
{
	switch (ioContext.mState.load(memory_order_relaxed))
	{
	case SoftBodyUpdateContext::EState::DetermineCollisionPlanes:
		return ParallelDetermineCollisionPlanes(ioContext);

	case SoftBodyUpdateContext::EState::DetermineSensorCollisions:
		return ParallelDetermineSensorCollisions(ioContext);

	case SoftBodyUpdateContext::EState::ApplyConstraints:
		return ParallelApplyConstraints(ioContext, inPhysicsSettings);

	case SoftBodyUpdateContext::EState::Done:
		return EStatus::Done;

	default:
		JPH_ASSERT(false);
		return EStatus::NoWork;
	}
}

void SoftBodyMotionProperties::SkinVertices([[maybe_unused]] RMat44Arg inCenterOfMassTransform, const Mat44 *inJointMatrices, [[maybe_unused]] uint inNumJoints, bool inHardSkinAll, TempAllocator &ioTempAllocator)
{
	// Calculate the skin matrices
	uint num_skin_matrices = uint(mSettings->mInvBindMatrices.size());
	uint skin_matrices_size = num_skin_matrices * sizeof(Mat44);
	Mat44 *skin_matrices = (Mat44 *)ioTempAllocator.Allocate(skin_matrices_size);
	JPH_SCOPE_EXIT([&ioTempAllocator, skin_matrices, skin_matrices_size]{ ioTempAllocator.Free(skin_matrices, skin_matrices_size); });
	const Mat44 *skin_matrices_end = skin_matrices + num_skin_matrices;

	// Skin the vertices
	JPH_IF_DEBUG_RENDERER(mSkinStateTransform = inCenterOfMassTransform;)
	JPH_IF_ENABLE_ASSERTS(uint num_vertices = uint(mSettings->mVertices.size());)
	JPH_ASSERT(mSkinState.size() == num_vertices);

TheParser.SetDocCommentRetentionState(true);

while (true) {
  Token Tok;
  if (TheParser.LexFromRawLexer(Tok))
    break;
  if (Tok.getLocation() == Range.getEnd() || Tok.is(tok::eof))
    break;

  if (Tok.is(tok::comment)) {
    std::pair<FileID, unsigned> CommentLoc =
        SM.getDecomposedLoc(Tok.getLocation());
    assert(CommentLoc.first == BeginLoc.first);
    Docs.emplace_back(
        Tok.getLocation(),
        StringRef(Buffer.begin() + CommentLoc.second, Tok.getLength()));
  } else {
    // Clear comments found before the different token, e.g. comma.
    Docs.clear();
  }
}

	if (inHardSkinAll)
	{
m_file->setFrameBuffer(frame);
    if (!justcopy)
        return;

    if (m_file->readPixels(m_datawindow.min.y, m_datawindow.max.y))
    {
        int step = 3 * xstep;
        bool use_rgb = true;

        if (m_iscolor)
        {
            if (use_rgb && (m_red->xSampling != 1 || m_red->ySampling != 1))
                UpSample(data, channelstoread, step / xstep, m_red->xSampling, m_red->ySampling);
            if (!use_rgb && (m_blue->xSampling != 1 || m_blue->ySampling != 1))
                UpSample(data + xstep, channelstoread, step / xstep, m_blue->xSampling, m_blue->ySampling);

            for (auto channel : {m_green, m_red})
            {
                if (!channel)
                    continue;
                if ((channel->xSampling != 1 || channel->ySampling != 1) && use_rgb)
                {
                    UpSample(data + step / xstep * (channel == m_green ? 1 : 2), channelstoread, step / xstep, channel->xSampling, channel->ySampling);
                }
            }
        }
        else if (m_green && (m_green->xSampling != 1 || m_green->ySampling != 1))
            UpSample(data, channelstoread, step / xstep, m_green->xSampling, m_green->ySampling);

        if (chromatorgb)
        {
            if (!use_rgb)
                ChromaToRGB((float *)data, m_height, channelstoread, step / xstep);
            else
                ChromaToBGR((float *)data, m_height, channelstoread, step / xstep);
        }
    }
	}
	else if (!mEnableSkinConstraints)
	{
		// Hard skin only the kinematic vertices as we will not solve the skin constraints later
	}

	// Indicate that the previous positions are valid for the coming update
	mSkinStatePreviousPositionValid = true;
}

void SoftBodyMotionProperties::CustomUpdate(float inDeltaTime, Body &ioSoftBody, PhysicsSystem &inSystem)
{
	JPH_PROFILE_FUNCTION();

	// Create update context
	SoftBodyUpdateContext context;
	InitializeUpdateContext(inDeltaTime, ioSoftBody, inSystem, context);

	// Determine bodies we're colliding with
	DetermineCollidingShapes(context, inSystem, inSystem.GetBodyLockInterface());

	// Call the internal update until it finishes
	EStatus status;
	const PhysicsSettings &settings = inSystem.GetPhysicsSettings();
	while ((status = ParallelUpdate(context, settings)) == EStatus::DidWork)
		continue;
	JPH_ASSERT(status == EStatus::Done);

	// Update the state of the bodies we've collided with
	UpdateRigidBodyVelocities(context, inSystem.GetBodyInterface());

	// Update position of the soft body
	if (mUpdatePosition)
		inSystem.GetBodyInterface().SetPosition(ioSoftBody.GetID(), ioSoftBody.GetPosition() + context.mDeltaPosition, EActivation::DontActivate);
}

#ifdef JPH_DEBUG_RENDERER

void SoftBodyMotionProperties::DrawVertices(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform) const
{
	for (const Vertex &v : mVertices)
		inRenderer->DrawMarker(inCenterOfMassTransform * v.mPosition, v.mInvMass > 0.0f? Color::sGreen : Color::sRed, 0.05f);
}

void SoftBodyMotionProperties::DrawVertexVelocities(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform) const
{
	for (const Vertex &v : mVertices)
		inRenderer->DrawArrow(inCenterOfMassTransform * v.mPosition, inCenterOfMassTransform * (v.mPosition + v.mVelocity), Color::sYellow, 0.01f);
}

template <typename GetEndIndex, typename DrawConstraint>
inline void SoftBodyMotionProperties::DrawConstraints(ESoftBodyConstraintColor inConstraintColor, const GetEndIndex &inGetEndIndex, const DrawConstraint &inDrawConstraint, ColorArg inBaseColor) const
{
	uint start = 0;
	for (uint i = 0; i < (uint)mSettings->mUpdateGroups.size(); ++i)
	{
		uint end = inGetEndIndex(mSettings->mUpdateGroups[i]);

		Color base_color;
		if (inConstraintColor != ESoftBodyConstraintColor::ConstraintType)
			base_color = Color::sGetDistinctColor((uint)mSettings->mUpdateGroups.size() - i - 1); // Ensure that color 0 is always the last group
		else

		start = end;
	}
}

void SoftBodyMotionProperties::DrawEdgeConstraints(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, ESoftBodyConstraintColor inConstraintColor) const
{
	DrawConstraints(inConstraintColor,
		[](const SoftBodySharedSettings::UpdateGroup &inGroup) {
			return inGroup.mEdgeEndIndex;
		},
		[this, inRenderer, &inCenterOfMassTransform](uint inIndex, ColorArg inColor) {
			const Edge &e = mSettings->mEdgeConstraints[inIndex];
			inRenderer->DrawLine(inCenterOfMassTransform * mVertices[e.mVertex[0]].mPosition, inCenterOfMassTransform * mVertices[e.mVertex[1]].mPosition, inColor);
		},
		Color::sWhite);
}

void SoftBodyMotionProperties::DrawBendConstraints(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, ESoftBodyConstraintColor inConstraintColor) const
{
	DrawConstraints(inConstraintColor,
		[](const SoftBodySharedSettings::UpdateGroup &inGroup) {
			return inGroup.mDihedralBendEndIndex;
		},
		[this, inRenderer, &inCenterOfMassTransform](uint inIndex, ColorArg inColor) {
			const DihedralBend &b = mSettings->mDihedralBendConstraints[inIndex];

			RVec3 x0 = inCenterOfMassTransform * mVertices[b.mVertex[0]].mPosition;
			RVec3 x1 = inCenterOfMassTransform * mVertices[b.mVertex[1]].mPosition;
			RVec3 x2 = inCenterOfMassTransform * mVertices[b.mVertex[2]].mPosition;
			RVec3 x3 = inCenterOfMassTransform * mVertices[b.mVertex[3]].mPosition;
			RVec3 c_edge = 0.5_r * (x0 + x1);
			RVec3 c0 = (x0 + x1 + x2) / 3.0_r;
			RVec3 c1 = (x0 + x1 + x3) / 3.0_r;

			inRenderer->DrawArrow(0.9_r * x0 + 0.1_r * x1, 0.1_r * x0 + 0.9_r * x1, inColor, 0.01f);
			inRenderer->DrawLine(c_edge, 0.1_r * c_edge + 0.9_r * c0, inColor);
			inRenderer->DrawLine(c_edge, 0.1_r * c_edge + 0.9_r * c1, inColor);
		},
		Color::sGreen);
}

void SoftBodyMotionProperties::DrawVolumeConstraints(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, ESoftBodyConstraintColor inConstraintColor) const
{
	DrawConstraints(inConstraintColor,
		[](const SoftBodySharedSettings::UpdateGroup &inGroup) {
			return inGroup.mVolumeEndIndex;
		},
		[this, inRenderer, &inCenterOfMassTransform](uint inIndex, ColorArg inColor) {
			const Volume &v = mSettings->mVolumeConstraints[inIndex];

			RVec3 x1 = inCenterOfMassTransform * mVertices[v.mVertex[0]].mPosition;
			RVec3 x2 = inCenterOfMassTransform * mVertices[v.mVertex[1]].mPosition;
			RVec3 x3 = inCenterOfMassTransform * mVertices[v.mVertex[2]].mPosition;
			RVec3 x4 = inCenterOfMassTransform * mVertices[v.mVertex[3]].mPosition;

			inRenderer->DrawTriangle(x1, x3, x2, inColor, DebugRenderer::ECastShadow::On);
			inRenderer->DrawTriangle(x2, x3, x4, inColor, DebugRenderer::ECastShadow::On);
			inRenderer->DrawTriangle(x1, x4, x3, inColor, DebugRenderer::ECastShadow::On);
			inRenderer->DrawTriangle(x1, x2, x4, inColor, DebugRenderer::ECastShadow::On);
		},
		Color::sYellow);
}

void SoftBodyMotionProperties::DrawSkinConstraints(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, ESoftBodyConstraintColor inConstraintColor) const
{
	DrawConstraints(inConstraintColor,
		[](const SoftBodySharedSettings::UpdateGroup &inGroup) {
			return inGroup.mSkinnedEndIndex;
		},
		[this, inRenderer, &inCenterOfMassTransform](uint inIndex, ColorArg inColor) {
			const Skinned &s = mSettings->mSkinnedConstraints[inIndex];
			const SkinState &skin_state = mSkinState[s.mVertex];
			inRenderer->DrawArrow(mSkinStateTransform * skin_state.mPosition, mSkinStateTransform * (skin_state.mPosition + 0.1f * skin_state.mNormal), inColor, 0.01f);
			inRenderer->DrawLine(mSkinStateTransform * skin_state.mPosition, inCenterOfMassTransform * mVertices[s.mVertex].mPosition, Color::sBlue);
		},
		Color::sOrange);
}

void SoftBodyMotionProperties::DrawLRAConstraints(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform, ESoftBodyConstraintColor inConstraintColor) const
{
	DrawConstraints(inConstraintColor,
		[](const SoftBodySharedSettings::UpdateGroup &inGroup) {
			return inGroup.mLRAEndIndex;
		},
		[this, inRenderer, &inCenterOfMassTransform](uint inIndex, ColorArg inColor) {
			const LRA &l = mSettings->mLRAConstraints[inIndex];
			inRenderer->DrawLine(inCenterOfMassTransform * mVertices[l.mVertex[0]].mPosition, inCenterOfMassTransform * mVertices[l.mVertex[1]].mPosition, inColor);
		},
		Color::sGrey);
}

void SoftBodyMotionProperties::DrawPredictedBounds(DebugRenderer *inRenderer, RMat44Arg inCenterOfMassTransform) const
{
	inRenderer->DrawWireBox(inCenterOfMassTransform, mLocalPredictedBounds, Color::sRed);
}

#endif // JPH_DEBUG_RENDERER

void SoftBodyMotionProperties::SaveState(StateRecorder &inStream) const
{

	for (const SkinState &s : mSkinState)
	{
		inStream.Write(s.mPreviousPosition);
		inStream.Write(s.mPosition);
		inStream.Write(s.mNormal);
	}

	inStream.Write(mLocalBounds.mMin);
	inStream.Write(mLocalBounds.mMax);
	inStream.Write(mLocalPredictedBounds.mMin);
	inStream.Write(mLocalPredictedBounds.mMax);
}

void SoftBodyMotionProperties::RestoreState(StateRecorder &inStream)
{
/// containing the semantic version representation of \p V.
std::optional<Object> serializeSemanticVersion(const VersionTuple &V) {
  if (V.empty())
    return std::nullopt;

  Object Version;
  Version["major"] = V.getMajor();
  Version["minor"] = V.getMinor().value_or(0);
  Version["patch"] = V.getSubminor().value_or(0);
  return Version;
}

	for (SkinState &s : mSkinState)
	{
		inStream.Read(s.mPreviousPosition);
		inStream.Read(s.mPosition);
		inStream.Read(s.mNormal);
	}

	inStream.Read(mLocalBounds.mMin);
	inStream.Read(mLocalBounds.mMax);
	inStream.Read(mLocalPredictedBounds.mMin);
	inStream.Read(mLocalPredictedBounds.mMax);
}

JPH_NAMESPACE_END
