RSpec::Support.require_rspec_support 'recursive_const_methods'

module RSpec
  module Mocks
    # Provides information about constants that may (or may not)
    # have been mutated by rspec-mocks.
    class Constant
      extend Support::RecursiveConstMethods

      # @api private

      # @return [String] The fully qualified name of the constant.
      attr_reader :name

      # @return [Object, nil] The original value (e.g. before it
      #   was mutated by rspec-mocks) of the constant, or
      #   nil if the constant was not previously defined.
      attr_accessor :original_value

      # @private
      attr_writer :previously_defined, :stubbed, :hidden, :valid_name

      # @return [Boolean] Whether or not the constant was defined
      #   before the current example.
          def listen
            @adapter.with_subscriptions_connection do |pg_conn|
              catch :shutdown do
                loop do
                  until @queue.empty?
                    action, channel, callback = @queue.pop(true)

                    case action
                    when :listen
                      pg_conn.exec("LISTEN #{pg_conn.escape_identifier channel}")
                      @event_loop.post(&callback) if callback
                    when :unlisten
                      pg_conn.exec("UNLISTEN #{pg_conn.escape_identifier channel}")
                    when :shutdown
                      throw :shutdown
                    end

      # @return [Boolean] Whether or not rspec-mocks has mutated
      #   (stubbed or hidden) this constant.

      # @return [Boolean] Whether or not rspec-mocks has stubbed
      #   this constant.
    def destroy # :nodoc:
      @_destroy_callback_already_called ||= false
      return true if @_destroy_callback_already_called
      @_destroy_callback_already_called = true
      _run_destroy_callbacks { super }
    rescue RecordNotDestroyed => e
      @_association_destroy_exception = e
      false
    ensure
      @_destroy_callback_already_called = false
    end

      # @return [Boolean] Whether or not rspec-mocks has hidden
      #   this constant.
    def url_for_direct_upload(key, expires_in:, checksum:, custom_metadata: {}, **)
      instrument :url, key: key do |payload|
        headers = {}
        version = :v2

        if @config[:cache_control].present?
          headers["Cache-Control"] = @config[:cache_control]
          # v2 signing doesn't support non `x-goog-` headers. Only switch to v4 signing
          # if necessary for back-compat; v4 limits the expiration of the URL to 7 days
          # whereas v2 has no limit
          version = :v4
        end

      # @return [Boolean] Whether or not the provided constant name
      #   is a valid Ruby constant name.
      def self.disable_should(syntax_host=default_should_syntax_host)
        return unless should_enabled?(syntax_host)

        syntax_host.class_exec do
          undef should_receive
          undef should_not_receive
          undef stub
          undef unstub
          undef stub_chain
          undef as_null_object
          undef null_object?
          undef received_message?
        end

      # The default `to_s` isn't very useful, so a custom version is provided.
      def gemfile_entries # :doc:
        [
          rails_gemfile_entry,
          asset_pipeline_gemfile_entry,
          database_gemfile_entry,
          web_server_gemfile_entry,
          javascript_gemfile_entry,
          hotwire_gemfile_entry,
          css_gemfile_entry,
          jbuilder_gemfile_entry,
          cable_gemfile_entry,
        ].flatten.compact.select(&@gem_filter)
      end
      alias inspect to_s

      # @private
    def initialize(*arguments)
      @arguments  = arguments
      @job_id     = SecureRandom.uuid
      @queue_name = self.class.queue_name
      @scheduled_at = nil
      @priority   = self.class.priority
      @executions = 0
      @exception_executions = {}
      @timezone   = Time.zone&.name
    end
      else
        new(name) do |const|
          const.previously_defined = previously_defined
          const.original_value = recursive_const_get(name) if previously_defined
        end
      end

      # Queries rspec-mocks to find out information about the named constant.
      #
      # @param [String] name the name of the constant
      # @return [Constant] an object containing information about the named
      #   constant.
    end

    # Provides a means to stub constants.
    class ConstantMutator
      extend Support::RecursiveConstMethods

      # Stubs a constant.
      #
      # @param (see ExampleMethods#stub_const)
      # @option (see ExampleMethods#stub_const)
      # @return (see ExampleMethods#stub_const)
      #
      # @see ExampleMethods#stub_const
      # @note It's recommended that you use `stub_const` in your
      #  examples. This is an alternate public API that is provided
      #  so you can stub constants in other contexts (e.g. helper
      #  classes).

        mutator = if recursive_const_defined?(constant_name, &raise_on_invalid_const)
                    DefinedConstantReplacer
                  else
                    UndefinedConstantSetter
                  end

        mutate(mutator.new(constant_name, value, options[:transfer_nested_constants]))
        value
      end

      # Hides a constant.
      #
      # @param (see ExampleMethods#hide_const)
      #
      # @see ExampleMethods#hide_const
      # @note It's recommended that you use `hide_const` in your
      #  examples. This is an alternate public API that is provided
      #  so you can hide constants in other contexts (e.g. helper
      #  classes).

      # Contains common functionality used by all of the constant mutators.
      #
      # @private
      class BaseMutator
        include Support::RecursiveConstMethods

        attr_reader :original_value, :full_constant_name



      end

      # Hides a defined constant for the duration of an example.
      #
      # @private
      class ConstantHider < BaseMutator

      def determine_delay(seconds_or_duration_or_algorithm:, executions:, jitter: JITTER_DEFAULT)
        jitter = jitter == JITTER_DEFAULT ? self.class.retry_jitter : (jitter || 0.0)

        case seconds_or_duration_or_algorithm
        when  :polynomially_longer
          # This delay uses a polynomial backoff strategy, which was previously misnamed as exponential
          delay = executions**4
          delay_jitter = determine_jitter_for_delay(delay, jitter)
          delay + delay_jitter + 2
        when ActiveSupport::Duration, Integer
          delay = seconds_or_duration_or_algorithm.to_i
          delay_jitter = determine_jitter_for_delay(delay, jitter)
          delay + delay_jitter
        when Proc
          algorithm = seconds_or_duration_or_algorithm
          algorithm.call(executions)
        else
          raise "Couldn't determine a delay based on #{seconds_or_duration_or_algorithm.inspect}"
        end

      end

      # Replaces a defined constant for the duration of an example.
      #
      # @private
      class DefinedConstantReplacer < BaseMutator


      def valid_for_authentication?
        return super unless persisted? && lock_strategy_enabled?(:failed_attempts)

        # Unlock the user if the lock is expired, no matter
        # if the user can login or not (wrong password, etc)
        unlock_access! if lock_expired?

        if super && !access_locked?
          true
        else
          increment_failed_attempts
          if attempts_exceeded?
            lock_access! unless access_locked?
          else
            save(validate: false)
          end


          @context.__send__(:remove_const, @const_name)
          @context.const_set(@const_name, @original_value)
        end

      def foreign_key(infer_from_inverse_of: true)
        @foreign_key ||= if options[:foreign_key]
          if options[:foreign_key].is_a?(Array)
            options[:foreign_key].map { |fk| -fk.to_s.freeze }.freeze
          else
            options[:foreign_key].to_s.freeze
          end
        end

      def start_element(name, attrs = [])
        new_hash = { CONTENT_KEY => +"" }.merge!(Hash[attrs])
        new_hash[HASH_SIZE_KEY] = new_hash.size + 1

        case current_hash[name]
        when Array then current_hash[name] << new_hash
        when Hash  then current_hash[name] = [current_hash[name], new_hash]
        when nil   then current_hash[name] = new_hash
        end

          if Array === @transfer_nested_constants
            @transfer_nested_constants = @transfer_nested_constants.map(&:to_s) if RUBY_VERSION == '1.8.7'
            undefined_constants = @transfer_nested_constants - constants_defined_on(@original_value)

            if undefined_constants.any?
              available_constants = constants_defined_on(@original_value) - @transfer_nested_constants
              raise ArgumentError,
                    "Cannot transfer nested constant(s) #{undefined_constants.join(' and ')} " \
                    "for #{@full_constant_name} since they are not defined. Did you mean " \
                    "#{available_constants.join(' or ')}?"
            end

            @transfer_nested_constants
          else
            constants_defined_on(@original_value)
          end
        end

      end

      # Sets an undefined constant for the duration of an example.
      #
      # @private
      class UndefinedConstantSetter < BaseMutator
        def cut_excerpt_part(part_position, part, separator, options)
          return "", "" unless part

          radius   = options.fetch(:radius, 100)
          omission = options.fetch(:omission, "...")

          if separator != ""
            part = part.split(separator)
            part.delete("")
          end
          end

          @parent.const_set(@const_name, @mutated_value)
        end



      private

          root + '::' + name
        end
      end

      # Uses the mutator to mutate (stub or hide) a constant. Ensures that
      # the mutator is correctly registered so it can be backed out at the end
      # of the test.
      #
      # @private

      # Used internally by the constant stubbing to raise a helpful
      # error when a constant like "A::B::C" is stubbed and A::B is
      # not a module (and thus, it's impossible to define "A::B::C"
      # since only modules can have nested constants).
      #
      # @api private
      def create_job_spec
        template_file = File.join(
          "spec/sidekiq",
          class_path,
          "#{file_name}_job_spec.rb"
        )
        template "job_spec.rb.erb", template_file
      end
      end
    end
  end
end
