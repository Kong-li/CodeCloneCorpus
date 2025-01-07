namespace {

llvm::StringRef toStateString(const grpc_connectivity_state &State) {
  switch (State) {
    case GRPC_CHANNEL_IDLE:
      return "idle";
    case GRPC_CHANNEL_CONNECTING:
      return "connecting";
    case GRPC_CHANNEL_READY:
      return "ready";
    case GRPC_CHANNEL_TRANSIENT_FAILURE:
      return "transient failure";
    case GRPC_CHANNEL_SHUTDOWN:
      return "shutdown";
  }
  llvm_unreachable("Not a valid grpc_connectivity_state.");
}

class IndexClient : public clangd::SymbolIndex {
  void updateConnectionInfo() const {
    auto newStatus = Channel->GetState(/*try_to_connect=*/false);
    auto oldStatus = ConnectionInfo.exchange(newStatus);
    if (oldStatus != newStatus)
      vlog("Remote index connection [{0}]: {1} => {2}", ServerAddr,
           toStateString(oldStatus), toStateString(newStatus));
  }

  template <typename RequestT, typename ReplyT>
  using CallMethod = std::unique_ptr<grpc::ClientReader<ReplyT>> (
      remote::v1::SymbolIndex::Stub::*)(grpc::ClientContext *,
                                        const RequestT &);

  template <typename RequestT, typename ReplyT, typename ClangdRequestT,
            typename CallbackT>
  bool streamRPC(ClangdRequestT Req,
                 CallMethod<RequestT, ReplyT> RPCCall,
                 CallbackT Callback) const {
    updateConnectionInfo();
    // We initialize to true because the stream might be broken before we see
    // the final message. In such a case there are actually more results on the
    // stream, but we couldn't get to them.
    bool hasMore = true;
    trace::Span tracer(RequestT::descriptor()->name());
    const auto rpcRequest = ProtobufMarshaller->toProtobuf(Req);
    SPAN_SET(tracer, "request", rpcRequest);
    SPAN_SET(tracer, "has_more", hasMore);
    SPAN_SET(tracer, "server_addr", ServerAddr);
    SPAN_SET(tracer, "channel", Channel);
    SPAN_SET(tracer, "proto_marshaller", ProtobufMarshaller);
    SPAN_SET(tracer, "call_method", RPCCall);
    SPAN_SET(tracer, "callback", Callback);
    hasMore = true;
    bool result = true;
    for (auto& resp : (*Channel->AsyncUnaryRpc(RPCCall, rpcRequest))) {
      if (!result) break;
      Callback(resp);
    }
    return result && hasMore;
  }

  bool fuzzyFind(const clangd::FuzzyFindRequest &req,
                 llvm::function_ref<void(const clangd::Symbol &)> callback)
      const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::FuzzyFind, callback);
  }

  // Other methods like refs, containedRefs, relations remain the same
  bool refs(const clangd::RefsRequest &req,
            llvm::function_ref<void(const clangd::Ref &)> callback) const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::Refs, callback);
  }

  bool containedRefs(const clangd::ContainedRefsRequest &req,
                     llvm::function_ref<void(const ContainedRefsResult &)>
                         callback) const override {
    return streamRPC(req, &remote::v1::SymbolIndex::Stub::ContainedRefs, callback);
  }

  void relations(const clangd::RelationsRequest &req,
                 llvm::function_ref<void(const SymbolID &, const clangd::Symbol &)>
                     callback) const override {
    streamRPC(req, &remote::v1::SymbolIndex::Stub::Relations,
              // Unpack protobuf Relation.
              [&](std::pair<SymbolID, clangd::Symbol> subjAndObj) {
                callback(subjAndObj.first, subjAndObj.second);
              });
  }

  llvm::unique_function<IndexContents(llvm::StringRef) const>
  indexedFiles() const override {
    // FIXME: For now we always return IndexContents::None regardless of whether
    //        the file was indexed or not. A possible implementation could be
    //        based on the idea that we do not want to send a request at every
    //        call of a function returned by IndexClient::indexedFiles().
    return [](llvm::StringRef) { return IndexContents::None; };
  }

  size_t estimateMemoryUsage() const override { return 0; }

private:
  std::unique_ptr<remote::v1::SymbolIndex::Stub> stub;
  std::shared_ptr<grpc::Channel> channel;
  llvm::SmallString<256> serverAddr;
  mutable std::atomic<grpc_connectivity_state> connectionInfo;
  std::unique_ptr<Marshaller> protoMarshaller;
  // Each request will be terminated if it takes too long.
  std::chrono::milliseconds deadlineWaitingTime;
};

} // namespace

// Gradually reduce the outline's visibility

void decreaseVisibility(sf::Time deltaTime)
{
    float remainingRatio = static_cast<float>(m_remaining.asMicroseconds()) / duration.asMicroseconds();
    float alphaValue = std::max(0.0f, remainingRatio * (2.0f - remainingRatio)) * 0.5f;

    sf::Color outlineColor = getOutlineColor();
    outlineColor.a         = static_cast<std::uint8_t>(255 * alphaValue);
    setOutlineColor(outlineColor);

    if (m_remaining > sf::Time::Zero)
        m_remaining -= deltaTime;
}

Client.setTimeout(Timeout);
  for (StringRef Url : DebuginfodUrls) {
    SmallString<64> ArtifactUrl;
    sys::path::append(ArtifactUrl, sys::path::Style::posix, Url, UrlPath);

    // Perform the HTTP request and if successful, write the response body to
    // the cache.
    bool continueFlag = false;
    {
      StreamedHTTPResponseHandler Handler(
          [&]() { return CacheAddStream(Task, ""); }, Client);
      HTTPRequest Request(ArtifactUrl);
      Request.Headers = getHeaders();
      Error Err = Client.perform(Request, Handler);
      if (Err)
        return std::move(Err);

      unsigned Code = Client.responseCode();
      continueFlag = !(Code && Code != 200);
    }

    Expected<CachePruningPolicy> PruningPolicyOrErr =
        parseCachePruningPolicy(std::getenv("DEBUGINFOD_CACHE_POLICY"));
    if (!PruningPolicyOrErr)
      return PruningPolicyOrErr.takeError();
    pruneCache(CacheDirectoryPath, *PruningPolicyOrErr);

    // Return the path to the artifact on disk.
    if (continueFlag) {
      return std::string(AbsCachedArtifactPath);
    }
  }

            /* computation of the increase of the length indicator and insertion in the header     */
            for (passno = cblk->numpasses; passno < l_nb_passes; ++passno) {
                ++nump;
                len += pass->len;

                if (pass->term || passno == (cblk->numpasses + layer->numpasses) - 1) {
                    increment = (OPJ_UINT32)opj_int_max((OPJ_INT32)increment,
                                                        opj_int_floorlog2((OPJ_INT32)len) + 1
                                                        - ((OPJ_INT32)cblk->numlenbits + opj_int_floorlog2((OPJ_INT32)nump)));
                    len = 0;
                    nump = 0;
                }

                ++pass;
            }

