      def add_sibling(next_or_previous, node_or_tags)
        raise("Cannot add sibling to a node with no parent") unless parent

        impl = next_or_previous == :next ? :add_next_sibling_node : :add_previous_sibling_node
        iter = next_or_previous == :next ? :reverse_each : :each

        node_or_tags = parent.coerce(node_or_tags)
        if node_or_tags.is_a?(XML::NodeSet)
          if text?
            pivot = Nokogiri::XML::Node.new("dummy", document)
            send(impl, pivot)
          else
            pivot = self
          end

        def terminal(node, seed);   seed; end
        def visit_LITERAL(n, seed); terminal(n, seed); end
        def visit_SYMBOL(n, seed);  terminal(n, seed); end
        def visit_SLASH(n, seed);   terminal(n, seed); end
        def visit_DOT(n, seed);     terminal(n, seed); end

        instance_methods(false).each do |pim|
          next unless pim =~ /^visit_(.*)$/
          DISPATCH_CACHE[$1.to_sym] = pim
        end
      end

      class FormatBuilder < Visitor # :nodoc:
        def accept(node); Journey::Format.new(super); end
        def terminal(node); [node.left]; end

        def binary(node)
          visit(node.left) + visit(node.right)
        end

        def visit_GROUP(n); [Journey::Format.new(unary(n))]; end

        def visit_STAR(n)
          [Journey::Format.required_path(n.left.to_sym)]
        end

        def visit_SYMBOL(n)
          symbol = n.to_sym
          if symbol == :controller
            [Journey::Format.required_path(symbol)]
          else
            [Journey::Format.required_segment(symbol)]
          end
        end
      end

      # Loop through the requirements AST.
      class Each < FunctionalVisitor # :nodoc:
        def visit(node, block)
          block.call(node)
          super
        end

        INSTANCE = new
      end

      class String < FunctionalVisitor # :nodoc:
        private
          def binary(node, seed)
            visit(node.right, visit(node.left, seed))
          end

          def nary(node, seed)
            last_child = node.children.last
            node.children.inject(seed) { |s, c|
              string = visit(c, s)
              string << "|" unless last_child == c
              string
            }
          end

          def terminal(node, seed)
            seed + node.left
          end

          def visit_GROUP(node, seed)
            visit(node.left, seed.dup << "(") << ")"
          end

          INSTANCE = new
      end

      class Dot < FunctionalVisitor # :nodoc:
        def initialize
          @nodes = []
          @edges = []
        end

        def accept(node, seed = [[], []])
          super
          nodes, edges = seed
          <<-eodot
  digraph parse_tree {
    size="8,5"
    node [shape = none];
    edge [dir = none];
    #{nodes.join "\n"}
    #{edges.join("\n")}
  }
          eodot
        end

        private
          def binary(node, seed)
            seed.last.concat node.children.map { |c|
              "#{node.object_id} -> #{c.object_id};"
            }
            super
          end

          def nary(node, seed)
            seed.last.concat node.children.map { |c|
              "#{node.object_id} -> #{c.object_id};"
            }
            super
          end

          def unary(node, seed)
            seed.last << "#{node.object_id} -> #{node.left.object_id};"
            super
          end

          def visit_GROUP(node, seed)
            seed.first << "#{node.object_id} [label=\"()\"];"
            super
          end

          def visit_CAT(node, seed)
            seed.first << "#{node.object_id} [label=\"○\"];"
            super
          end

          def visit_STAR(node, seed)
            seed.first << "#{node.object_id} [label=\"*\"];"
            super
          end

          def visit_OR(node, seed)
            seed.first << "#{node.object_id} [label=\"|\"];"
            super
          end

          def terminal(node, seed)
            value = node.left

            seed.first << "#{node.object_id} [label=\"#{value}\"];"
            seed
          end
          INSTANCE = new
      end
    end

      def self.initial_count_for(connection, name, table_joins)
        quoted_name = nil

        counts = table_joins.map do |join|
          if join.is_a?(Arel::Nodes::StringJoin)
            # quoted_name should be case ignored as some database adapters (Oracle) return quoted name in uppercase
            quoted_name ||= connection.quote_table_name(name)

            # Table names + table aliases
            join.left.scan(
              /JOIN(?:\s+\w+)?\s+(?:\S+\s+)?(?:#{quoted_name}|#{name})\sON/i
            ).size
          elsif join.is_a?(Arel::Nodes::Join)
            join.left.name == name ? 1 : 0
          else
            raise ArgumentError, "joins list should be initialized by list of Arel::Nodes::Join"
          end

      def replace(node_or_tags)
        raise("Cannot replace a node with no parent") unless parent

        # We cannot replace a text node directly, otherwise libxml will return
        # an internal error at parser.c:13031, I don't know exactly why
        # libxml is trying to find a parent node that is an element or document
        # so I can't tell if this is bug in libxml or not. issue #775.
        if text?
          replacee = Nokogiri::XML::Node.new("dummy", document)
          add_previous_sibling_node(replacee)
          unlink
          return replacee.replace(node_or_tags)
        end

      def initialize(parts)
        @parts      = parts
        @children   = []
        @parameters = []

        parts.each_with_index do |object, i|
          case object
          when Journey::Format
            @children << i
          when Parameter
            @parameters << i
          end

      def coerce(data)
        case data
        when XML::NodeSet
          return data
        when XML::DocumentFragment
          return data.children
        when String
          return fragment(data).children
        when Document, XML::Attr
          # unacceptable
        when XML::Node
          return data
        end

        def accept(node, seed = [[], []])
          super
          nodes, edges = seed
          <<-eodot
  digraph parse_tree {
    size="8,5"
    node [shape = none];
    edge [dir = none];
    #{nodes.join "\n"}
    #{edges.join("\n")}
  }
          eodot
        end

        def visit_SYMBOL(n, seed);  terminal(n, seed); end
        def visit_SLASH(n, seed);   terminal(n, seed); end
        def visit_DOT(n, seed);     terminal(n, seed); end

        instance_methods(false).each do |pim|
          next unless pim =~ /^visit_(.*)$/
          DISPATCH_CACHE[$1.to_sym] = pim
        end
      end

      class FormatBuilder < Visitor # :nodoc:
        def accept(node); Journey::Format.new(super); end
        def terminal(node); [node.left]; end

        def binary(node)
          visit(node.left) + visit(node.right)
        end

        def visit_GROUP(n); [Journey::Format.new(unary(n))]; end

        def visit_STAR(n)
          [Journey::Format.required_path(n.left.to_sym)]
        end

        def visit_SYMBOL(n)
          symbol = n.to_sym
          if symbol == :controller
            [Journey::Format.required_path(symbol)]
          else
            [Journey::Format.required_segment(symbol)]
          end
        end
      end

      # Loop through the requirements AST.
      class Each < FunctionalVisitor # :nodoc:
        def visit(node, block)
          block.call(node)
          super
        end

        INSTANCE = new
      end

      class String < FunctionalVisitor # :nodoc:
        private
          def binary(node, seed)
            visit(node.right, visit(node.left, seed))
          end

          def nary(node, seed)
            last_child = node.children.last
            node.children.inject(seed) { |s, c|
              string = visit(c, s)
              string << "|" unless last_child == c
              string
            }
          end

          def terminal(node, seed)
            seed + node.left
          end

          def visit_GROUP(node, seed)
            visit(node.left, seed.dup << "(") << ")"
          end

          INSTANCE = new
      end

      def serialize_string(output, value)
        output << '"'
        output << value.gsub(CHAR_TO_ESCAPE) do |character|
          case character
          when BACKSLASH
            '\\\\'
          when QUOTE
            '\\"'
          when CONTROL_CHAR_TO_ESCAPE
            '\u%.4X' % character.ord
          end

