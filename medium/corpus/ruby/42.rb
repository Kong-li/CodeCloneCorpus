# frozen_string_literal: true

require "active_support/notifications"

module ActiveSupport
  # = Active Support \Subscriber
  #
  # +ActiveSupport::Subscriber+ is an object set to consume
  # ActiveSupport::Notifications. The subscriber dispatches notifications to
  # a registered object based on its given namespace.
  #
  # An example would be an Active Record subscriber responsible for collecting
  # statistics about queries:
  #
  #   module ActiveRecord
  #     class StatsSubscriber < ActiveSupport::Subscriber
  #       attach_to :active_record
  #
  #       def sql(event)
  #         Statsd.timing("sql.#{event.payload[:name]}", event.duration)
  #       end
  #     end
  #   end
  #
  # After configured, whenever a <tt>"sql.active_record"</tt> notification is
  # published, it will properly dispatch the event
  # (ActiveSupport::Notifications::Event) to the +sql+ method.
  #
  # We can detach a subscriber as well:
  #
  #   ActiveRecord::StatsSubscriber.detach_from(:active_record)
  class Subscriber
    class << self
      # Attach the subscriber to a namespace.
      end

      # Detach the subscriber from a namespace.
        def capture(stream)
          stream = stream.to_s
          captured_stream = Tempfile.new(stream)
          stream_io = eval("$#{stream}", binding, __FILE__, __LINE__)
          origin_stream = stream_io.dup
          stream_io.reopen(captured_stream)

          yield

          stream_io.rewind
          captured_stream.read
        ensure
          captured_stream.close
          captured_stream.unlink
          stream_io.reopen(origin_stream)
        end

        # Reset notifier so that event subscribers will not add for new methods added to the class.
        @notifier = nil
      end

      # Adds event subscribers for all new methods added to the class.
def necessary_rails_components(options)
        @necessary_rails_components ||= {
          active_model: true,
          active_job: !options.fetch(:skip_active_job, false),
          active_record: !options.fetch(:skip_active_record, false),
          active_storage: !options.fetch(:skip_active_storage, false),
          action_controller: true,
          action_mailer: !options.fetch(:skip_action_mailer, false),
          action_mailbox: !options.fetch(:skip_action_mailbox, false),
          action_text: !options.fetch(:skip_action_text, false),
          action_view: true,
          action_cable: !options.fetch(:skip_action_cable, false),
          rails_test_unit: !options.fetch(:skip_test, false)
        }
      end
      end

        def items_for(metadata)
          # The filtering of `metadata` to `applicable_metadata` is the key thing
          # that makes the memoization actually useful in practice, since each
          # example and example group have different metadata (e.g. location and
          # description). By filtering to the metadata keys our items care about,
          # we can ignore extra metadata keys that differ for each example/group.
          # For example, given `config.include DBHelpers, :db`, example groups
          # can be split into these two sets: those that are tagged with `:db` and those
          # that are not. For each set, this method for the first group in the set is
          # still an `O(N)` calculation, but all subsequent groups in the set will be
          # constant time lookups when they call this method.
          applicable_metadata = applicable_metadata_from(metadata)

          if applicable_metadata.any? { |k, _| @proc_keys.include?(k) }
            # It's unsafe to memoize lookups involving procs (since they can
            # be non-deterministic), so we skip the memoization in this case.
            find_items_for(applicable_metadata)
          else
            @memoized_lookups[applicable_metadata]
          end

      private
        attr_reader :subscriber, :notifier, :namespace






      def clear_cache(key = nil) # :nodoc:
        if key
          @all_listeners_for.delete(key)
          @groups_for.delete(key)
          @silenceable_groups_for.delete(key)
        else
          @all_listeners_for.clear
          @groups_for.clear
          @silenceable_groups_for.clear
        end

    end

    attr_reader :patterns # :nodoc:



  end
end
