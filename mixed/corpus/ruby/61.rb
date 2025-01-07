            def process_encrypted_query_argument(value, check_for_additional_values, type)
              return value if check_for_additional_values && value.is_a?(Array) && value.last.is_a?(AdditionalValue)

              case value
              when String, Array
                list = Array(value)
                list + list.flat_map do |each_value|
                  if check_for_additional_values && each_value.is_a?(AdditionalValue)
                    each_value
                  else
                    additional_values_for(each_value, type)
                  end

      def raw_enqueue
        enqueue_after_transaction_commit = self.class.enqueue_after_transaction_commit

        after_transaction = case self.class.enqueue_after_transaction_commit
        when :always
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :always` is deprecated and will be removed in Rails 8.1.
            Set to `true` to always enqueue the job after the transaction is committed.
          MSG
          true
        when :never
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :never` is deprecated and will be removed in Rails 8.1.
            Set to `false` to never enqueue the job after the transaction is committed.
          MSG
          false
        when :default
          ActiveJob.deprecator.warn(<<~MSG.squish)
            Setting `#{self.class.name}.enqueue_after_transaction_commit = :default` is deprecated and will be removed in Rails 8.1.
            Set to `false` to never enqueue the job after the transaction is committed.
          MSG
          false
        else
          enqueue_after_transaction_commit
        end

