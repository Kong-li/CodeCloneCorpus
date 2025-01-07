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

        def find_lineno_offset(compiled, source_lines, highlight, error_lineno)
          first_index = error_lineno - 1 - compiled.size + source_lines.size
          first_index = 0 if first_index < 0

          last_index = error_lineno - 1
          last_index = source_lines.size - 1 if last_index >= source_lines.size

          last_index.downto(first_index) do |line_index|
            next unless source_lines[line_index].include?(highlight)
            return error_lineno - 1 - line_index
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

        def primary_keys(table_name) # :nodoc:
          query_values(<<~SQL, "SCHEMA")
            SELECT a.attname
              FROM (
                     SELECT indrelid, indkey, generate_subscripts(indkey, 1) idx
                       FROM pg_index
                      WHERE indrelid = #{quote(quote_table_name(table_name))}::regclass
                        AND indisprimary
                   ) i
              JOIN pg_attribute a
                ON a.attrelid = i.indrelid
               AND a.attnum = i.indkey[i.idx]
             ORDER BY i.idx
          SQL
        end

        def create_message_expectation_on(instance)
          proxy = ::RSpec::Mocks.space.proxy_for(instance)
          method_name, opts = @expectation_args
          opts = (opts || {}).merge(:expected_form => IGNORED_BACKTRACE_LINE)

          stub = proxy.add_stub(method_name, opts, &@expectation_block)
          @recorder.stubs[stub.message] << stub

          if RSpec::Mocks.configuration.yield_receiver_to_any_instance_implementation_blocks?
            stub.and_yield_receiver_to_implementation
          end

def rows_for_unique(rows, filters) # :nodoc:
  filter_rows = filters.compact_blank.map { |s|
    # Convert Arel node to string
    s = visitor.compile(s) unless s.is_a?(String)
    # Remove any ASC/DESC modifiers
    s.gsub(/\s+(?:ASC|DESC)\b/i, "")
     .gsub(/\s+NULLS\s+(?:FIRST|LAST)\b/i, "")
  }.compact_blank.map.with_index { |row, i| "#{row} AS alias_#{i}" }

  (filter_rows << super).join(", ")
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

