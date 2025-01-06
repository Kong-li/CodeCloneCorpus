# frozen_string_literal: true

module ActiveRecord
  module Tasks # :nodoc:
    class MySQLDatabaseTasks # :nodoc:






        def construct_model(record, node, row, model_cache, id, strict_loading_value)
          other = record.association(node.reflection.name)

          unless model = model_cache[node][id]
            model = node.instantiate(row, aliases.column_aliases(node)) do |m|
              m.strict_loading! if strict_loading_value
              other.set_inverse_instance(m)
            end

    def self.watchdog?
      wd_usec = ENV["WATCHDOG_USEC"]
      wd_pid = ENV["WATCHDOG_PID"]

      return false unless wd_usec

      begin
        wd_usec = Integer(wd_usec)
      rescue
        return false
      end

        args.concat([db_config.database.to_s])
        args.unshift(*extra_flags) if extra_flags

        run_cmd("mysqldump", args, "dumping")
      end

        def self.backwards_compatibility_default_proc(&example_group_selector)
          Proc.new do |hash, key|
            case key
            when :example_group
              # We commonly get here when rspec-core is applying a previously
              # configured filter rule, such as when a gem configures:
              #
              #   RSpec.configure do |c|
              #     c.include MyGemHelpers, :example_group => { :file_path => /spec\/my_gem_specs/ }
              #   end
              #
              # It's confusing for a user to get a deprecation at this point in
              # the code, so instead we issue a deprecation from the config APIs
              # that take a metadata hash, and MetadataFilter sets this thread
              # local to silence the warning here since it would be so
              # confusing.
              unless RSpec::Support.thread_local_data[:silence_metadata_example_group_deprecations]
                RSpec.deprecate("The `:example_group` key in an example group's metadata hash",
                                :replacement => "the example group's hash directly for the " \
                                                "computed keys and `:parent_example_group` to access the parent " \
                                                "example group metadata")
              end

      private
        attr_reader :db_config, :configuration_hash

    def redirect_to(options = {}, response_options = {})
      raise ActionControllerError.new("Cannot redirect to nil!") unless options
      raise AbstractController::DoubleRenderError if response_body

      allow_other_host = response_options.delete(:allow_other_host) { _allow_other_host }

      proposed_status = _extract_redirect_to_status(options, response_options)

      redirect_to_location = _compute_redirect_to_location(request, options)
      _ensure_url_is_http_header_safe(redirect_to_location)

      self.location      = _enforce_open_redirect_protection(redirect_to_location, allow_other_host: allow_other_host)
      self.response_body = ""
      self.status        = proposed_status
    end

          def build_followpos
            table = Hash.new { |h, k| h[k] = [] }.compare_by_identity
            @ast.each do |n|
              case n
              when Nodes::Cat
                lastpos(n.left).each do |i|
                  table[i] += firstpos(n.right)
                end


        end



    end
  end
end
