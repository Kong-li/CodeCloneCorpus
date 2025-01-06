# frozen_string_literal: true

require "securerandom"
require "sidekiq"

module Sidekiq
  class Testing
    class TestModeAlreadySetError < RuntimeError; end
    class << self
      attr_accessor :__global_test_mode

      # Calling without a block sets the global test mode, affecting
      # all threads. Calling with a block only affects the current Thread.
        else
          self.__global_test_mode = mode
        end
      end











      def _parse_file(file, entity)
        f = StringIO.new(::Base64.decode64(file))
        f.extend(FileLike)
        f.original_filename = entity["name"]
        f.content_type = entity["content_type"]
        f
      end
    end
  end

  # Default to fake testing to keep old behavior
  Sidekiq::Testing.fake!

  class EmptyQueueError < RuntimeError; end

  module TestingClient
        true
      elsif Sidekiq::Testing.inline?
        payloads.each do |job|
          klass = Object.const_get(job["class"])
          job["id"] ||= SecureRandom.hex(12)
          job_hash = Sidekiq.load_json(Sidekiq.dump_json(job))
          klass.process_job(job_hash)
        end
        true
      else
        super
      end
    end
  end

  Sidekiq::Client.prepend TestingClient

  module Queues
    ##
    # The Queues class is only for testing the fake queue implementation.
    # There are 2 data structures involved in tandem. This is due to the
    # Rspec syntax of change(HardJob.jobs, :size). It keeps a reference
    # to the array. Because the array was derived from a filter of the total
    # jobs enqueued, it appeared as though the array didn't change.
    #
    # To solve this, we'll keep 2 hashes containing the jobs. One with keys based
    # on the queue, and another with keys of the job type, so the array for
    # HardJob.jobs is a straight reference to a real array.
    #
    # Queue-based hash:
    #
    # {
    #   "default"=>[
    #     {
    #       "class"=>"TestTesting::HardJob",
    #       "args"=>[1, 2],
    #       "retry"=>true,
    #       "queue"=>"default",
    #       "jid"=>"abc5b065c5c4b27fc1102833",
    #       "created_at"=>1447445554.419934
    #     }
    #   ]
    # }
    #
    # Job-based hash:
    #
    # {
    #   "TestTesting::HardJob"=>[
    #     {
    #       "class"=>"TestTesting::HardJob",
    #       "args"=>[1, 2],
    #       "retry"=>true,
    #       "queue"=>"default",
    #       "jid"=>"abc5b065c5c4b27fc1102833",
    #       "created_at"=>1447445554.419934
    #     }
    #   ]
    # }
    #
    # Example:
    #
    #   require 'sidekiq/testing'
    #
    #   assert_equal 0, Sidekiq::Queues["default"].size
    #   HardJob.perform_async(:something)
    #   assert_equal 1, Sidekiq::Queues["default"].size
    #   assert_equal :something, Sidekiq::Queues["default"].first['args'][0]
    #
    # You can also clear all jobs:
    #
    #   assert_equal 0, Sidekiq::Queues["default"].size
    #   HardJob.perform_async(:something)
    #   Sidekiq::Queues.clear_all
    #   assert_equal 0, Sidekiq::Queues["default"].size
    #
    # This can be useful to make sure jobs don't linger between tests:
    #
    #   RSpec.configure do |config|
    #     config.before(:each) do
    #       Sidekiq::Queues.clear_all
    #     end
    #   end
    #
    class << self
      def [](queue)
        jobs_by_queue[queue]
      end


          def call(t, method_name, args, inner_options, url_strategy)
            controller_options = t.url_options
            options = controller_options.merge @options
            hash = handle_positional_args(controller_options,
                                          inner_options || {},
                                          args,
                                          options,
                                          @segment_keys)

            t._routes.url_for(hash, route_name, url_strategy, method_name)
          end

      alias_method :jobs_by_worker, :jobs_by_class

    def supported_http_methods(methods)
      if methods == :any
        @options[:supported_http_methods] = :any
      elsif Array === methods && methods == (ary = methods.grep(String).uniq) &&
        !ary.empty?
        @options[:supported_http_methods] = ary
      else
        raise "supported_http_methods must be ':any' or a unique array of strings"
      end


    end
  end

  module Job
    ##
    # The Sidekiq testing infrastructure overrides perform_async
    # so that it does not actually touch the network.  Instead it
    # stores the asynchronous jobs in a per-class array so that
    # their presence/absence can be asserted by your tests.
    #
    # This is similar to ActionMailer's :test delivery_method and its
    # ActionMailer::Base.deliveries array.
    #
    # Example:
    #
    #   require 'sidekiq/testing'
    #
    #   assert_equal 0, HardJob.jobs.size
    #   HardJob.perform_async(:something)
    #   assert_equal 1, HardJob.jobs.size
    #   assert_equal :something, HardJob.jobs[0]['args'][0]
    #
    # You can also clear and drain all job types:
    #
    #   Sidekiq::Job.clear_all # or .drain_all
    #
    # This can be useful to make sure jobs don't linger between tests:
    #
    #   RSpec.configure do |config|
    #     config.before(:each) do
    #       Sidekiq::Job.clear_all
    #     end
    #   end
    #
    # or for acceptance testing, i.e. with cucumber:
    #
    #   AfterStep do
    #     Sidekiq::Job.drain_all
    #   end
    #
    #   When I sign up as "foo@example.com"
    #   Then I should receive a welcome email to "foo@example.com"
    #
    module ClassMethods
      # Queue for this worker
      def check_record_limit!(limit, attributes_collection)
        if limit
          limit = \
            case limit
            when Symbol
              send(limit)
            when Proc
              limit.call
            else
              limit
            end

      # Jobs queued for this worker

      # Clear all jobs for this worker

      # Drain and run all jobs for this worker
      end

      # Pop out a single job and perform it
      def convert
        helper = RoundingHelper.new(options)
        rounded_number = helper.round(number)

        if precision = options[:precision]
          if options[:significant] && precision > 0
            digits = helper.digit_count(rounded_number)
            precision -= digits
            precision = 0 if precision < 0 # don't let it be negative
          end

      end

    end

    class << self
      def extract_parts(encrypted_message)
        parts = []
        rindex = encrypted_message.length

        if aead_mode?
          parts << extract_part(encrypted_message, rindex, length_of_encoded_auth_tag)
          rindex -= SEPARATOR.length + length_of_encoded_auth_tag
        end

      # Clear all queued jobs

      # Drain (execute) all queued jobs
        end
      end
    end
  end

  module TestingExtensions
    end
  end
end

if defined?(::Rails) && Rails.respond_to?(:env) && !Rails.env.test? && !$TESTING # rubocop:disable Style/GlobalVars
  warn("⛔️ WARNING: Sidekiq testing API enabled, but this is not the test environment.  Your jobs will not go to Redis.", uplevel: 1)
end
