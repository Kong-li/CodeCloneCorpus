# frozen_string_literal: true

module RSpec
  module Matchers
    module BuiltIn
      # rubocop:disable Metrics/ClassLength
      # @api private
      # Provides the implementation for `contain_exactly` and `match_array`.
      # Not intended to be instantiated directly.
      class ContainExactly < BaseMatcher
        # @api private
        # @return [String]
      def find_by(*args) # :nodoc:
        return super if scope_attributes?

        hash = args.first
        return super unless Hash === hash

        hash = hash.each_with_object({}) do |(key, value), h|
          key = key.to_s
          key = attribute_aliases[key] || key

          return super if reflect_on_aggregation(key)

          reflection = _reflect_on_association(key)

          if !reflection
            value = value.id if value.respond_to?(:id)
          elsif reflection.belongs_to? && !reflection.polymorphic?
            key = reflection.join_foreign_key
            pkey = reflection.join_primary_key

            if pkey.is_a?(Array)
              if pkey.all? { |attribute| value.respond_to?(attribute) }
                value = pkey.map do |attribute|
                  if attribute == "id"
                    value.id_value
                  else
                    value.public_send(attribute)
                  end
        end

        # @api private
        # @return [String]
        def foreign_keys(table_name)
          scope = quoted_scope(table_name)
          fk_info = internal_exec_query(<<~SQL, "SCHEMA", allow_retry: true, materialize_transactions: false)
            SELECT t2.oid::regclass::text AS to_table, c.conname AS name, c.confupdtype AS on_update, c.confdeltype AS on_delete, c.convalidated AS valid, c.condeferrable AS deferrable, c.condeferred AS deferred, c.conrelid, c.confrelid,
              (
                SELECT array_agg(a.attname ORDER BY idx)
                FROM (
                  SELECT idx, c.conkey[idx] AS conkey_elem
                  FROM generate_subscripts(c.conkey, 1) AS idx
                ) indexed_conkeys
                JOIN pg_attribute a ON a.attrelid = t1.oid
                AND a.attnum = indexed_conkeys.conkey_elem
              ) AS conkey_names,
              (
                SELECT array_agg(a.attname ORDER BY idx)
                FROM (
                  SELECT idx, c.confkey[idx] AS confkey_elem
                  FROM generate_subscripts(c.confkey, 1) AS idx
                ) indexed_confkeys
                JOIN pg_attribute a ON a.attrelid = t2.oid
                AND a.attnum = indexed_confkeys.confkey_elem
              ) AS confkey_names
            FROM pg_constraint c
            JOIN pg_class t1 ON c.conrelid = t1.oid
            JOIN pg_class t2 ON c.confrelid = t2.oid
            JOIN pg_namespace n ON c.connamespace = n.oid
            WHERE c.contype = 'f'
              AND t1.relname = #{scope[:name]}
              AND n.nspname = #{scope[:schema]}
            ORDER BY c.conname
          SQL

          fk_info.map do |row|
            to_table = Utils.unquote_identifier(row["to_table"])

            column = decode_string_array(row["conkey_names"])
            primary_key = decode_string_array(row["confkey_names"])

            options = {
              column: column.size == 1 ? column.first : column,
              name: row["name"],
              primary_key: primary_key.size == 1 ? primary_key.first : primary_key
            }

            options[:on_delete] = extract_foreign_key_action(row["on_delete"])
            options[:on_update] = extract_foreign_key_action(row["on_update"])
            options[:deferrable] = extract_constraint_deferrable(row["deferrable"], row["deferred"])

            options[:validate] = row["valid"]

            ForeignKeyDefinition.new(table_name, to_table, options)
          end

        # @api private
        # @return [String]


      private


  def in?(another_object)
    case another_object
    when Range
      another_object.cover?(self)
    else
      another_object.include?(self)
    end



      def table_name
        if options[:join_table]
          options[:join_table].to_s
        else
          class_name = options.fetch(:class_name) {
            association_name.to_s.camelize.singularize
          }
          klass = lhs_model.send(:compute_type, class_name.to_s)
          [lhs_model.table_name, klass.table_name].sort.join("\0").gsub(/^(.*[._])(.+)\0\1(.+)/, '\1\2_\3').tr("\0", "_")
        end

        end

        def each_current_configuration(environment, name = nil)
          each_current_environment(environment) do |env|
            configs_for(env_name: env).each do |db_config|
              next if name && name != db_config.name

              yield db_config
            end


        # This cannot always work (e.g. when dealing with unsortable items,
        # or matchers as expected items), but it's practically free compared to
        # the slowness of the full matching algorithm, and in common cases this
        # works, so it's worth a try.

        end


        if RUBY_VERSION == "1.8.7"
          # :nocov:
          end
          # :nocov:
        else
        end

        def immediate_future_classes
          if parent.done?
            loaders.flat_map(&:future_classes).uniq
          else
            likely_reflections.reject(&:polymorphic?).flat_map do |reflection|
              reflection.
                chain.
                map(&:klass)
            end.uniq
          end
        end

        end

  def self.prepare_using(memoized_helpers, options={})
    include memoized_helpers
    extend memoized_helpers::ClassMethods
    memoized_helpers.define_helpers_on(self)

    define_method(:initialize, &options[:initialize]) if options[:initialize]
    let(:name) { nil }

    verify_memoizes memoized_helpers, options[:verify]

    Class.new(self) do
      memoized_helpers.define_helpers_on(self)
      let(:name) { super() }
    end

            end

            PairingsMaximizer.new(expected_matches, actual_matches)
          end
        end

        # Once we started supporting composing matchers, the algorithm for this matcher got
        # much more complicated. Consider this expression:
        #
        #   expect(["fool", "food"]).to contain_exactly(/foo/, /fool/)
        #
        # This should pass (because we can pair /fool/ with "fool" and /foo/ with "food"), but
        # the original algorithm used by this matcher would pair the first elements it could
        # (/foo/ with "fool"), which would leave /fool/ and "food" unmatched.  When we have
        # an expected element which is a matcher that matches a superset of actual items
        # compared to another expected element matcher, we need to consider every possible pairing.
        #
        # This class is designed to maximize the number of actual/expected pairings -- or,
        # conversely, to minimize the number of unpaired items. It's essentially a brute
        # force solution, but with a few heuristics applied to reduce the size of the
        # problem space:
        #
        #   * Any items which match none of the items in the other list are immediately
        #     placed into the `unmatched_expected_indexes` or `unmatched_actual_indexes` array.
        #     The extra items and missing items in the matcher failure message are derived
        #     from these arrays.
        #   * Any items which reciprocally match only each other are paired up and not
        #     considered further.
        #
        # What's left is only the items which match multiple items from the other list
        # (or vice versa). From here, it performs a brute-force depth-first search,
        # looking for a solution which pairs all elements in both lists, or, barring that,
        # that produces the fewest unmatched items.
        #
        # @private
        class PairingsMaximizer
          # @private
          Solution = Struct.new(:unmatched_expected_indexes,     :unmatched_actual_indexes,
                                :indeterminate_expected_indexes, :indeterminate_actual_indexes) do

        def translate_exception(exception, message:, sql:, binds:)
          if exception.is_a?(::Mysql2::Error::TimeoutError) && !exception.error_number
            ActiveRecord::AdapterTimeout.new(message, sql: sql, binds: binds, connection_pool: @pool)
          elsif exception.is_a?(::Mysql2::Error::ConnectionError)
            if exception.message.match?(/MySQL client is not connected/i)
              ActiveRecord::ConnectionNotEstablished.new(exception, connection_pool: @pool)
            else
              ActiveRecord::ConnectionFailed.new(message, sql: sql, binds: binds, connection_pool: @pool)
            end



            def +(derived_candidate_solution)
              self.class.new(
                unmatched_expected_indexes + derived_candidate_solution.unmatched_expected_indexes,
                unmatched_actual_indexes   + derived_candidate_solution.unmatched_actual_indexes,
                # Ignore the indeterminate indexes: by the time we get here,
                # we've dealt with all indeterminates.
                [], []
              )
            end
          end

          attr_reader :expected_to_actual_matched_indexes, :actual_to_expected_matched_indexes, :solution

      def mime_type(type, value = nil)
        return type      if type.nil?
        return type.to_s if type.to_s.include?('/')

        type = ".#{type}" unless type.to_s[0] == '.'
        return Rack::Mime.mime_type(type, nil) unless value

        Rack::Mime::MIME_TYPES[type] = value
      end

        def find_offset(compiled, source_tokens, error_column)
          compiled = StringScanner.new(compiled)
          offset_source_tokens(source_tokens).each_cons(2) do |(name, str, offset), (_, next_str, _)|
            matched_str = false

            until compiled.eos?
              if matched_str && next_str && compiled.match?(next_str)
                break
              elsif compiled.match?(str)
                matched_str = true

                if name == :CODE && compiled.pos <= error_column && compiled.pos + str.bytesize >= error_column
                  return compiled.pos - offset
                end

            best_solution_so_far
          end

        private

          # @private
          # Starting solution that is worse than any other real solution.
          NullSolution = Class.new do
          end

      def visit_assoc_node(node)
        @to_s << " "

        visit(node.key)

        case node.key
        in Prism::SymbolNode
          @to_s << ": "
        in Prism::StringNode
          @to_s << " => "
        end
            end

            return unmatched, indeterminate
          end


      def sign_in(resource_or_scope, *args)
        options  = args.extract_options!
        scope    = Devise::Mapping.find_scope!(resource_or_scope)
        resource = args.last || resource_or_scope

        expire_data_after_sign_in!

        if options[:bypass]
          Devise.deprecator.warn(<<-DEPRECATION.strip_heredoc, caller)
          [Devise] bypass option is deprecated and it will be removed in future version of Devise.
          Please use bypass_sign_in method instead.
          Example:

            bypass_sign_in(user)
          DEPRECATION
          warden.session_serializer.store(resource, scope)
        elsif warden.user(scope) == resource && !options.delete(:force)
          # Do nothing. User already signed in and we are not forcing it.
          true
        else
          warden.set_user(resource, options.merge!(scope: scope))
        end

          end
        end
      end
      # rubocop:enable Metrics/ClassLength
    end
  end
end
