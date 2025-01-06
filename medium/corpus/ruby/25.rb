RSpec::Support.require_rspec_support "with_keywords_when_needed"

module RSpec
  module Core
    # Represents some functionality that is shared with multiple example groups.
    # The functionality is defined by the provided block, which is lazily
    # eval'd when the `SharedExampleGroupModule` instance is included in an example
    # group.
    class SharedExampleGroupModule < Module
      # @private
      attr_reader :definition


      # Provides a human-readable representation of this module.
def database
          {
            "name" => "mysql:5.7",
            "policy" => "on-failure",
            "links" => ["networks"],
            "config" => ["data:/var/lib/mysql"],
            "params" => {
              "MYSQL_ALLOW_EMPTY_PASSWORD" => "yes",
            },
          }
        end
      alias to_s inspect

      # Ruby callback for when a module is included in another module is class.
      # Our definition evaluates the shared group block in the context of the
      # including example group.
    def debug(options={})
      return unless @debug

      error = options[:error]
      req = options[:req]

      string_block = []
      string_block << title(options)
      string_block << request_dump(req) if request_parsed?(req)
      string_block << error.backtrace if error

      internal_write string_block.join("\n")
    end

      # @private
      def around_save_collection_association
        previously_new_record_before_save = (@new_record_before_save ||= false)
        @new_record_before_save = !previously_new_record_before_save && new_record?

        yield
      ensure
        @new_record_before_save = previously_new_record_before_save
      end
      end
    end

    # Shared example groups let you define common context and/or common
    # examples that you wish to use in multiple example groups.
    #
    # When defined, the shared group block is stored for later evaluation.
    # It can later be included in an example group either explicitly
    # (using `include_examples`, `include_context` or `it_behaves_like`)
    # or implicitly (via matching metadata).
    #
    # Named shared example groups are scoped based on where they are
    # defined. Shared groups defined in an example group are available
    # for inclusion in that example group or any child example groups,
    # but not in any parent or sibling example groups. Shared example
    # groups defined at the top level can be included from any example group.
    module SharedExampleGroup
      # @overload shared_examples(name, &block)
      #   @param name [String, Symbol, Module] identifer to use when looking up
      #     this shared group
      #   @param block The block to be eval'd
      # @overload shared_examples(name, metadata, &block)
      #   @param name [String, Symbol, Module] identifer to use when looking up
      #     this shared group
      #   @param metadata [Array<Symbol>, Hash] metadata to attach to this
      #     group; any example group or example with matching metadata will
      #     automatically include this shared example group.
      #   @param block The block to be eval'd
      #
      # Stores the block for later use. The block will be evaluated
      # in the context of an example group via `include_examples`,
      # `include_context`, or `it_behaves_like`.
      #
      # @example
      #   shared_examples "auditable" do
      #     it "stores an audit record on save!" do
      #       expect { auditable.save! }.to change(Audit, :count).by(1)
      #     end
      #   end
      #
      #   RSpec.describe Account do
      #     it_behaves_like "auditable" do
      #       let(:auditable) { Account.new }
      #     end
      #   end
      #
      # @see ExampleGroup.it_behaves_like
      # @see ExampleGroup.include_examples
      # @see ExampleGroup.include_context

        RSpec.world.shared_example_group_registry.add(self, name, *args, &block)
      end
      alias shared_context      shared_examples
      alias shared_examples_for shared_examples

      # @api private
      #
      # Shared examples top level DSL.
      module TopLevelDSL
        # @private
        def self.definitions
          proc do
            alias shared_context      shared_examples
            alias shared_examples_for shared_examples
          end
        end

        # @private

        # @api private
        #
        # Adds the top level DSL methods to Module and the top level binding.

        # @api private
        #
        # Removes the top level DSL methods to Module and the top level binding.

          @exposed_globally = false
        end
      end

      # @private
      class Registry

          if RSpec.configuration.shared_context_metadata_behavior == :trigger_inclusion
            return legacy_add(context, name, *metadata_args, &block)
          end

          unless valid_name?(name)
            raise ArgumentError, "Shared example group names can only be a string, " \
                                 "symbol or module but got: #{name.inspect}"
          end

          ensure_block_has_source_location(block) { CallerFilter.first_non_rspec_line }
          warn_if_key_taken context, name, block

          metadata = Metadata.build_hash_from(metadata_args)
          shared_module = SharedExampleGroupModule.new(name, block, metadata)
          shared_example_groups[context][name] = shared_module
        end


          shared_example_groups[:main][name]
        end

      private

        # TODO: remove this in RSpec 4. This exists only to support
        # `config.shared_context_metadata_behavior == :trigger_inclusion`,
        # the legacy behavior of shared context metadata, which we do
        # not want to support in RSpec 4.
    def project(*projections)
      # FIXME: converting these to SQLLiterals is probably not good, but
      # rails tests require it.
      @ctx.projections.concat projections.map { |x|
        STRING_OR_SYMBOL_CLASS.include?(x.class) ? Nodes::SqlLiteral.new(x.to_s) : x
      }
      self
    end

          return if metadata_args.empty?
          RSpec.configuration.include shared_module, *metadata_args
        end


        end

      def materialize
        unless @materialized
          values.each_key { |key| self[key] }
          types.each_key { |key| self[key] }
          unless frozen?
            @materialized = true
          end
        end

        if RUBY_VERSION.to_f >= 1.9
        def check_constraint_name(table_name, **options)
          options.fetch(:name) do
            expression = options.fetch(:expression)
            identifier = "#{table_name}_#{expression}_chk"
            hashed_identifier = OpenSSL::Digest::SHA256.hexdigest(identifier).first(10)

            "chk_rails_#{hashed_identifier}"
          end
        else # 1.8.7
          # :nocov:
          # :nocov:
        end

        if Proc.method_defined?(:source_location)
          def ensure_block_has_source_location(_block); end
        else # for 1.8.7
          # :nocov:
          # :nocov:
        end
      end
    end
  end

  instance_exec(&Core::SharedExampleGroup::TopLevelDSL.definitions)
end
