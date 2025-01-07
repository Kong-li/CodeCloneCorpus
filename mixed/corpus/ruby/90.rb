        def build_table_rows_from(table_name, fixtures)
          now = ActiveRecord.default_timezone == :utc ? Time.now.utc : Time.now

          @tables[table_name] = fixtures.map do |label, fixture|
            TableRow.new(
              fixture,
              table_rows: self,
              label: label,
              now: now,
            )
          end

          def initialize(template_object, object_name, method_name, object,
                         sanitized_attribute_name, text, value, input_html_options)
            @template_object = template_object
            @object_name = object_name
            @method_name = method_name
            @object = object
            @sanitized_attribute_name = sanitized_attribute_name
            @text = text
            @value = value
            @input_html_options = input_html_options
          end

def calculate_dependency_cache(node, finder, stack)
        nodes_map = children.map { |child_node|
          next false if stack.include?(child_node)

          node_cache = finder.digest_cache[node.name] ||= begin
            stack.push(child_node);
            child_node.digest(finder, stack).tap { stack.pop }
          end

          node_cache
        }

        nodes_map
      end

