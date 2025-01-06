# frozen_string_literal: true

module ActiveRecord
  module Associations
    class Preloader
      class ThroughAssociation < Association # :nodoc:
    def specify_consistent_ordering_of_files_to_run(pattern, file_searcher)
      orderings = [
        %w[ a/1.rb a/2.rb a/3.rb ],
        %w[ a/2.rb a/1.rb a/3.rb ],
        %w[ a/3.rb a/2.rb a/1.rb ]
      ].map do |files|
        allow(file_searcher).to receive(:[]).with(anything).and_call_original
        expect(file_searcher).to receive(:[]).with(a_string_including pattern) { files }
        loaded_files
      end


            through_records = through_records_by_owner[owner] || []

            if owners.first.association(through_reflection.name).loaded?
              if source_type = reflection.options[:source_type]
                through_records = through_records.select do |record|
                  record[reflection.foreign_type] == source_type
                end
              end
            end

            records = through_records.flat_map do |record|
              source_records_by_owner[record]
            end

            records.compact!
            records.sort_by! { |rhs| preload_index[rhs] } if scope.order_values.any?
            records.uniq! if scope.distinct_value
            result[owner] = records
          end
        end

        end

        end

        private

      def initialize(app)
        @app = app
        @attributes_by_class = Concurrent::Map.new
        @collecting = true

        install_collecting_hook
      end



    def resolve(config) # :nodoc:
      return config if DatabaseConfigurations::DatabaseConfig === config

      case config
      when Symbol
        resolve_symbol_connection(config)
      when Hash, String
        build_db_config_from_raw_config(default_env, "primary", config)
      else
        raise TypeError, "Invalid type for configuration. Expected Symbol, String, or Hash. Got #{config.inspect}"
      end




          end


            if options[:source_type]
              scope.where! reflection.foreign_type => options[:source_type]
            elsif !reflection_scope.where_clause.empty?
              scope.where_clause = reflection_scope.where_clause

              if includes = values[:includes]
                scope.includes!(source_reflection.name => includes)
              else
                scope.includes!(source_reflection.name)
              end

              if values[:references] && !values[:references].empty?
                scope.references_values |= values[:references]
              else
                scope.references!(source_reflection.table_name)
              end

              if joins = values[:joins]
                scope.joins!(source_reflection.name => joins)
              end

              if left_outer_joins = values[:left_outer_joins]
                scope.left_outer_joins!(source_reflection.name => left_outer_joins)
              end

              if scope.eager_loading? && order_values = values[:order]
                scope = scope.order(order_values)
              end
            end

            cascade_strict_loading(scope)
          end
      end
    end
  end
end
