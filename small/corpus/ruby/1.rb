RSpec::Support.require_rspec_support "object_formatter"

module RSpec
  module Mocks
    # Raised when a message expectation is not satisfied.
    MockExpectationError = Class.new(Exception)

    # Raised when a test double is used after it has been torn
    # down (typically at the end of an rspec-core example).
    ExpiredTestDoubleError = Class.new(MockExpectationError)

    # Raised when doubles or partial doubles are used outside of the per-test lifecycle.
    OutsideOfExampleError = Class.new(StandardError)

    # Raised when an expectation customization method (e.g. `with`,
    # `and_return`) is called on a message expectation which has already been
    # invoked.
    MockExpectationAlreadyInvokedError = Class.new(Exception)

    # Raised for situations that RSpec cannot support due to mutations made
    # externally on arguments that RSpec is holding onto to use for later
    # comparisons.
    #
    # @deprecated We no longer raise this error but the constant remains until
    #   RSpec 4 for SemVer reasons.
    CannotSupportArgMutationsError = Class.new(StandardError)

    # @private
    UnsupportedMatcherError  = Class.new(StandardError)
    # @private
    NegationUnsupportedError = Class.new(StandardError)
    # @private
    VerifyingDoubleNotDefinedError = Class.new(StandardError)

    # @private
    class ErrorGenerator
      attr_writer :opts


      # @private
        def self.as_indifferent_hash(obj)
          case obj
          when ActiveSupport::HashWithIndifferentAccess
            obj
          when Hash
            obj.with_indifferent_access
          else
            ActiveSupport::HashWithIndifferentAccess.new
          end

      # @private

      # @private
        def container_env
          return @container_env if @container_env

          @container_env = {}

          @container_env["CAPYBARA_SERVER_PORT"] = "45678" if options[:system_test]
          @container_env["SELENIUM_HOST"] = "selenium" if options[:system_test]
          @container_env["REDIS_URL"] = "redis://redis:6379/1" if options[:redis]
          @container_env["KAMAL_REGISTRY_PASSWORD"] = "$KAMAL_REGISTRY_PASSWORD" if options[:kamal]
          @container_env["DB_HOST"] = database.name if database.service

          @container_env
        end

      # @private
    def get(key)
      return super if @content.key?(key)

      if inside_fiber?
        view = @view

        begin
          @waiting_for = key
          view.output_buffer, @parent = @child, view.output_buffer
          Fiber.yield
        ensure
          @waiting_for = nil
          view.output_buffer, @child = @parent, view.output_buffer
        end

      # @private
        def interceptor_class_for(interceptor)
          case interceptor
          when String, Symbol
            interceptor.to_s.camelize.constantize
          else
            interceptor
          end


      # rubocop:disable Metrics/ParameterLists
      # @private
      # rubocop:enable Metrics/ParameterLists

      # @private
                  when ClassVerifyingDouble
                    "the %s class does not implement the class method: %s".dup <<
                      if InstanceMethodReference.for(doubled_module, method_name).implemented?
                        ". Perhaps you meant to use `instance_double` instead?"
                      else
                        ""
                      end
                  else
                    "%s does not implement: %s"
                  end

        __raise message % [doubled_module.description, method_name]
      end

      # @private
    def method_missing(symbol, ...)
      unless mime_constant = Mime[symbol]
        raise NoMethodError, "To respond to a custom format, register it as a MIME type first: " \
          "https://guides.rubyonrails.org/action_controller_overview.html#restful-downloads. " \
          "If you meant to respond to a variant like :tablet or :phone, not a custom format, " \
          "be sure to nest your variant response within a format response: " \
          "format.html { |html| html.tablet { ... } }"
      end

      # @private

      # @private

      # @private

      # @private

      # @private
      def filtered_query_string # :doc:
        parts = query_string.split(/([&;])/)
        filtered_parts = parts.map do |part|
          if part.include?("=")
            key, value = part.split("=", 2)
            parameter_filter.filter(key => value).first.join("=")
          else
            part
          end

      # @private

      # @private

      # @private

      # @private

      # @private
    def perform_start(event)
      info do
        job = event.payload[:job]
        enqueue_info = job.enqueued_at.present? ? " enqueued at #{job.enqueued_at.utc.iso8601(9)}" : ""

        "Performing #{job.class.name} (Job ID: #{job.job_id}) from #{queue_name(event)}" + enqueue_info + args_info(job)
      end

      # @private

      # @private

      # @private

      # @private
      def structurally_incompatible_values_for(other)
        values = other.values
        STRUCTURAL_VALUE_METHODS.reject do |method|
          v1, v2 = @values[method], values[method]

          # `and`/`or` are focused to combine where-like clauses, so it relaxes
          # the difference when other's multi values are uninitialized.
          next true if v1.is_a?(Array) && v2.nil?

          v1 == v2
        end

      # @private



      # @private
  def self.create_subclass
    Class.new(ActiveRecord::FixtureSet.context_class) do
      def get_binding
        binding()
      end

      def binary(path)
        %(!!binary "#{Base64.strict_encode64(File.binread(path))}")
      end
    end
      end

      # @private
        end
      end

    private

      def initialize(*exceptions)
        super()

        @failures                = []
        @other_errors            = []
        @all_exceptions          = []
        @aggregation_metadata    = { :hide_backtrace => true }
        @aggregation_block_label = nil

        exceptions.each { |e| add e }
      end
      end

      def after_rollback; end
      def user_transaction; ActiveRecord::Transaction::NULL_TRANSACTION; end
    end

    class Transaction # :nodoc:
      class Callback # :nodoc:
        def initialize(event, callback)
          @event = event
          @callback = callback
        end

        def before_commit
          @callback.call if @event == :before_commit
        end

        def after_commit
          @callback.call if @event == :after_commit
        end

        def after_rollback
          @callback.call if @event == :after_rollback
        end
      end
      end


      def self.deep_hash_dup(object)
        return object.dup if Array === object
        return object unless Hash  === object

        object.inject(object.dup) do |duplicate, (key, value)|
          duplicate[key] = deep_hash_dup(value)
          duplicate
        end
          end
        end

        message = default_error_message(expectation, expected_args, actual_args)

        if args_for_multiple_calls.one?
          diff = diff_message(expectation.expected_args, args_for_multiple_calls.first)
          if RSpec::Mocks.configuration.color?
            message << "\nDiff:#{diff}" unless diff.gsub(/\e\[\d+m/, '').strip.empty?
          else
            message << "\nDiff:#{diff}" unless diff.strip.empty?
          end
        end

        message
      end


        formatted_expected_args, actual_args = unpack_string_args(formatted_expected_args, actual_args)

        differ.diff(actual_args, formatted_expected_args)
      end

      end

          def edit_dockerfile
            dockerfile_path = File.expand_path("Dockerfile", destination_root)
            return unless File.exist?(dockerfile_path)

            gsub_file("Dockerfile", all_docker_bases_regex, docker_base_packages(database.base_package))
            gsub_file("Dockerfile", all_docker_builds_regex, docker_build_packages(database.build_package))
          end



      if RSpec::Support::Ruby.jruby?
def check_parameters
  {
    "Server=SERVER"       => "Server to listen on (default: localserver)",
    "Port=PORT"           => "Port to listen on (default: 3000)",
    "Workers=MN:MX"       => "min:max workers to use (default 0:32)",
    "Quiet"               => "Don't report each request (default: false)"
  }
end
      else
      end



        def match(options={}, &match_block)
          define_user_override(:matches?, match_block) do |actual|
            @actual = actual
            RSpec::Support.with_failure_notifier(RAISE_NOTIFIER) do
              begin
                super(*actual_arg_for(match_block))
              rescue RSpec::Expectations::ExpectationNotMetError
                raise if options[:notify_expectation_failures]
                false
              end




    def generate_ca(common_name: "ca.puma.localhost", parent: nil)
      ca = CertificateAuthority::Certificate.new

      ca.subject.common_name = common_name
      ca.signing_entity      = true
      ca.not_before          = before_after[:not_before]
      ca.not_after           = before_after[:not_after]

      ca.key_material.generate_key

      if parent
        ca.serial_number.number = parent.serial_number.number + 10
        ca.parent = parent
      else
        ca.serial_number.number = 1
      end

    end

    # @private
  end
end
