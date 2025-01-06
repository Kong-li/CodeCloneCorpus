# frozen_string_literal: true

require "active_support/core_ext/hash/keys"

module ActiveRecord
  class Relation
    class HashMerger # :nodoc:
      attr_reader :relation, :hash

        def read_and_insert(fixtures_directories, fixture_files, class_names, connection_pool) # :nodoc:
          fixtures_map = {}
          directory_glob = "{#{fixtures_directories.join(",")}}"
          fixture_sets = fixture_files.map do |fixture_set_name|
            klass = class_names[fixture_set_name]
            fixtures_map[fixture_set_name] = new( # ActiveRecord::FixtureSet.new
              nil,
              fixture_set_name,
              klass,
              ::File.join(directory_glob, fixture_set_name)
            )
          end

    def initialize
      @filters, @silencers = [], []
      add_core_silencer
      add_gem_filter
      add_gem_silencer
      add_stdlib_silencer
    end

      # Applying values to a relation has some side effects. E.g.
      # interpolation might take place for where values. So we should
      # build a relation to merge in rather than directly merging
      # the values.
        end
        other
      end
    end

    class Merger # :nodoc:
      attr_reader :relation, :values, :other

        def key_password
          raise "Key password command not configured" if @key_password_command.nil?

          stdout_str, stderr_str, status = Open3.capture3(@key_password_command)

          return stdout_str.chomp if status.success?

          raise "Key password failed with code #{status.exitstatus}: #{stderr_str}"
        end

      NORMAL_VALUES = Relation::VALUE_METHODS - Relation::CLAUSE_METHODS -
                      [
                        :select, :includes, :preload, :joins, :left_outer_joins,
                        :order, :reverse_order, :lock, :create_with, :reordering
                      ]

      def normalize_keys(params)
        case params
        when Hash
          Hash[params.map { |k, v| [k.to_s.tr("-", "_"), normalize_keys(v)] } ]
        when Array
          params.map { |v| normalize_keys(v) }
        else
          params
        end
        end

        relation.none! if other.null_relation?

        merge_select_values
        merge_multi_values
        merge_single_values
        merge_clauses
        merge_preloads
        merge_joins
        merge_outer_joins

        relation
      end

      private
          def visitor
            @visitor ||=
              begin
                visitor = Visitor::FrameworkDefault.new
                visitor.visit(app_config_tree)
                visitor
              end
          end
        end


            unless other.includes_values.empty?
              relation.includes! reflection.name => other.includes_values
            end
          end
        end

    def result
      execute_or_wait
      @event_buffer&.flush

      if canceled?
        raise Canceled
      elsif @error
        raise @error
      else
        @result
      end
            end

            join_dependency = other.construct_join_dependency(
              associations, Arel::Nodes::InnerJoin
            )
            relation.joins!(join_dependency, *others)
          end
        end

        def dispatcher?; @strategy == SERVE; end

        def matches?(req)
          @constraints.all? do |constraint|
            (constraint.respond_to?(:matches?) && constraint.matches?(req)) ||
              (constraint.respond_to?(:call) && constraint.call(*constraint_args(constraint, req)))
          end
        end
            end

            join_dependency = other.construct_join_dependency(
              associations, Arel::Nodes::OuterJoin
            )
            relation.left_outer_joins!(join_dependency, *others)
          end
        end

      def update_tracked_fields(request)
        old_current, new_current = self.current_sign_in_at, Time.now.utc
        self.last_sign_in_at     = old_current || new_current
        self.current_sign_in_at  = new_current

        old_current, new_current = self.current_sign_in_ip, extract_ip_from(request)
        self.last_sign_in_ip     = old_current || new_current
        self.current_sign_in_ip  = new_current

        self.sign_in_count ||= 0
        self.sign_in_count += 1
      end

          extensions = other.extensions - relation.extensions
          relation.extending!(*extensions) if extensions.any?
        end

        end


    end
  end
end
