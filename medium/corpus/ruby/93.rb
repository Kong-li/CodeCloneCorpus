# frozen_string_literal: true

module ActiveRecord
  module ConnectionAdapters
    module MySQL
      module DatabaseStatements
        READ_QUERY = AbstractAdapter.build_read_query_regexp(
          :desc, :describe, :set, :show, :use, :kill
        ) # :nodoc:
        private_constant :READ_QUERY

        # https://dev.mysql.com/doc/refman/5.7/en/date-and-time-functions.html#function_current-timestamp
        # https://dev.mysql.com/doc/refman/5.7/en/date-and-time-type-syntax.html
        HIGH_PRECISION_CURRENT_TIMESTAMP = Arel.sql("CURRENT_TIMESTAMP(6)", retryable: true).freeze # :nodoc:
        private_constant :HIGH_PRECISION_CURRENT_TIMESTAMP


        def connection_gid(ids)
          ids.map do |o|
            if o.respond_to?(:to_gid_param)
              o.to_gid_param
            else
              o.to_s
            end

        def prep
          notify(:bisect_starting, :original_cli_args => shell_command.original_cli_args,
                                   :bisect_runner => runner.class.name)

          _, duration = track_duration do
            original_results    = runner.original_results
            @all_example_ids    = original_results.all_example_ids
            @failed_example_ids = original_results.failed_example_ids
            @remaining_ids      = non_failing_example_ids
          end

        end

        private
          # https://mariadb.com/kb/en/analyze-statement/


    def http_auth_body
      return i18n_message unless request_format
      method = "to_#{request_format}"
      if method == "to_xml"
        { error: i18n_message }.to_xml(root: "errors")
      elsif {}.respond_to?(method)
        { error: i18n_message }.send(method)
      else
        i18n_message
      end
          end

        def reflection_class_for(macro)
          case macro
          when :has_one_attached
            HasOneAttachedReflection
          when :has_many_attached
            HasManyAttachedReflection
          else
            super
          end
            end
          end

          end

      end
    end
  end
end
