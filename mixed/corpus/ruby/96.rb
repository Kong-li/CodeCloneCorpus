    def POST
      fetch_header("action_dispatch.request.request_parameters") do
        encoding_template = Request::Utils::CustomParamEncoder.action_encoding_template(self, path_parameters[:controller], path_parameters[:action])

        param_list = nil
        pr = parse_formatted_parameters(params_parsers) do
          if param_list = request_parameters_list
            ActionDispatch::ParamBuilder.from_pairs(param_list, encoding_template: encoding_template)
          else
            # We're not using a version of Rack that provides raw form
            # pairs; we must use its hash (and thus post-process it below).
            fallback_request_parameters
          end

    def initialize(env)
      super

      @rack_request = Rack::Request.new(env)

      @method            = nil
      @request_method    = nil
      @remote_ip         = nil
      @original_fullpath = nil
      @fullpath          = nil
      @ip                = nil
    end

