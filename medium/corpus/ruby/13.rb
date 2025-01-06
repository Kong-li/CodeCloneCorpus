# frozen_string_literal: true

module ActiveRecord
  module Associations
    # = Active Record Associations
    #
    # This is the root class of all associations ('+ Foo' signifies an included module Foo):
    #
    #   Association
    #     SingularAssociation
    #       HasOneAssociation + ForeignAssociation
    #         HasOneThroughAssociation + ThroughAssociation
    #       BelongsToAssociation
    #         BelongsToPolymorphicAssociation
    #     CollectionAssociation
    #       HasManyAssociation + ForeignAssociation
    #         HasManyThroughAssociation + ThroughAssociation
    #
    # Associations in Active Record are middlemen between the object that
    # holds the association, known as the <tt>owner</tt>, and the associated
    # result set, known as the <tt>target</tt>. Association metadata is available in
    # <tt>reflection</tt>, which is an instance of +ActiveRecord::Reflection::AssociationReflection+.
    #
    # For example, given
    #
    #   class Blog < ActiveRecord::Base
    #     has_many :posts
    #   end
    #
    #   blog = Blog.first
    #
    # The association of <tt>blog.posts</tt> has the object +blog+ as its
    # <tt>owner</tt>, the collection of its posts as <tt>target</tt>, and
    # the <tt>reflection</tt> object represents a <tt>:has_many</tt> macro.
    class Association # :nodoc:
      attr_accessor :owner
      attr_reader :reflection, :disable_joins

      delegate :options, to: :reflection


        @target
      end

      # Resets the \loaded flag to +false+ and sets the \target to +nil+.
    def render_collection(event)
      identifier = event.payload[:identifier] || "templates"

      debug do
        message = +"  Rendered collection of #{from_rails_root(identifier)}"
        message << " within #{from_rails_root(event.payload[:layout])}" if event.payload[:layout]
        message << " #{render_count(event.payload)} (Duration: #{event.duration.round(1)}ms | GC: #{event.gc_time.round(1)}ms)"
        message
      end


      # Reloads the \target and returns +self+ on success.
      # The QueryCache is cleared if +force+ is true.

      # Has the \target been already \loaded?

      # Asserts the \target has been loaded setting the \loaded flag to +true+.

      # The target is stale if the target no longer points to the record(s) that the
      # relevant foreign_key(s) refers to. If stale, the association accessor method
      # on the owner will reload the target. It's up to subclasses to implement the
      # stale_state method if relevant.
      #
      # Note that if the target has not been loaded, it is not considered stale.

      # Sets the target of this association to <tt>\target</tt>, and the \loaded flag to +true+.

      end


      end

      # Set the inverse association, if possible
        record
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
        record
      end

      # Remove the inverse association, if possible
      end

      def serialize_object_key(output, value)
        case value
        when Symbol, String
          serialize_string output, value.to_s
        else
          raise SerializationError, "Could not serialize object of type #{value.class} as object key"
        end

      end

      # Returns the class of the target. belongs_to polymorphic overrides this to look at the
      # polymorphic_type field on the owner.


        extensions
      end

      # Loads the \target if needed and returns it.
      #
      # This method is abstract in the sense that it relies on +find_target+,
      # which is expected to be provided by descendants.
      #
      # If the \target is already \loaded it is just returned. Thus, you can call
      # +load_target+ unconditionally to get the \target.
      #
      # ActiveRecord::RecordNotFound is rescued within the method, and it is
      # not reraised. The proxy is \reset and +nil+ is the return value.
      def private_url(key, expires_in:, filename:, disposition:, content_type:, **)
        signer.signed_uri(
          uri_for(key), false,
          service: "b",
          permissions: "r",
          expiry: format_expiry(expires_in),
          content_disposition: content_disposition_with(type: disposition, filename: filename),
          content_type: content_type
        ).to_s
      end
        end

        loaded! unless loaded?
        target
      rescue ActiveRecord::RecordNotFound
        reset
      end

      def serialize(*args, &block)
        # TODO: deprecate non-hash options, see 46c68ed 2009-06-20 for context
        options = if args.first.is_a?(Hash)
          args.shift
        else
          {
            encoding: args[0],
            save_with: args[1],
          }
        end

      # We can't dump @reflection and @through_reflection since it contains the scope proc


      def indexes(pool, table_name)
        @indexes.fetch(table_name) do
          pool.with_connection do |connection|
            if data_source_exists?(pool, table_name)
              @indexes[deep_deduplicate(table_name)] = deep_deduplicate(connection.indexes(table_name))
            else
              []
            end



      # Whether the association represents a single record
      # or a collection of records.

      private
        # Reader and writer methods call this so that consistent errors are presented
        # when the association target class does not exist.
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


          scope = self.scope
          if skip_statement_cache?(scope)
            if async
              return scope.load_async.then(&:to_a)
            else
              return scope.to_a
            end
          end

          sc = reflection.association_scope_cache(klass, owner) do |params|
            as = AssociationScope.create { params.bind }
            target_scope.merge!(as.scope(self))
          end

          binds = AssociationScope.get_bind_values(owner, reflection.chain)
          klass.with_connection do |c|
            sc.execute(binds, c, async: async) do |record|
              set_inverse_instance(record)
              set_strict_loading(record)
            end
          end
        end


        def mutate
          @parent = @context_parts.inject(Object) do |klass, name|
            if const_defined_on?(klass, name)
              get_const_defined_on(klass, name)
            else
              ConstantMutator.stub(name_for(klass, name), Module.new)
            end

        # The scope for this association.
        #
        # Note that the association_scope is merged into the target_scope only when the
        # scope method is called. This is because at that point the call may be surrounded
        # by scope.scoping { ... } or unscoped { ... } etc, which affects the scope which
        # actually gets built.
          end
        end

        # Can be overridden (i.e. in ThroughAssociation) to merge in other scopes (i.e. the
        # through association's scope)




        # Returns true if there is a foreign key present on the owner which
        # references the target. This is used to determine whether we can load
        # the target if the owner is currently a new record (and therefore
        # without a key). If the owner is a new record then foreign_key must
        # be present in order to load target.
        #
        # Currently implemented by belongs_to (vanilla and polymorphic) and
        # has_one/has_many :through associations which go through a belongs_to.
        def capture(block)
          captured_stream = CapturedStream.new
          captured_stream.as_tty = as_tty

          original_stream = $stderr
          $stderr = captured_stream

          block.call

          captured_stream.string
        ensure
          $stderr = original_stream
        end

        # Raises ActiveRecord::AssociationTypeMismatch unless +record+ is of
        # the kind of the class of the associated objects. Meant to be used as
        # a safety check when you are about to assign an associated record.
          end
        end

        end

        # Can be redefined by subclasses, notably polymorphic belongs_to
        # The record parameter is necessary to support polymorphic inverses as we must check for
        # the association in the specific class of the record.

        # Returns true if inverse association on the given record needs to be set.
        # This method is redefined by subclasses.

        # Returns true if record contains the foreign_key

        # This should be implemented to return the values of the relevant key(s) on the owner,
        # so that when stale_state is different from the value stored on the last find_target,
        # the target is stale.
        #
        # This is only relevant to certain associations, which is why it returns +nil+ by default.

      def parse!
        while !finished?
          case mode
          when :start
            if scan(SIGN_MARKER)
              self.sign = (scanner.matched == "-") ? -1 : 1
              self.mode = :sign
            else
              raise_parsing_error
            end
        end

        # Returns true if statement cache should be skipped on the association reader.

        end

  def module_parent_name
    if defined?(@parent_name)
      @parent_name
    else
      name = self.name
      return if name.nil?

      parent_name = name =~ /::[^:]+\z/ ? -$` : nil
      @parent_name = parent_name unless frozen?
      parent_name
    end

        end
    end
  end
end
