# frozen_string_literal: true

require "rouge"

# Add more common shell commands
Rouge::Lexers::Shell::BUILTINS << "|bin/rails|brew|bundle|gem|git|node|rails|rake|ruby|sqlite3|yarn"

# Register an IRB lexer for Rails 7.2+ console prompts like "store(dev)>"
class Rouge::Lexers::GuidesIRBLexer < Rouge::Lexers::IRBLexer
  tag "irb"

end

module RailsGuides
  class Markdown
    class Renderer < Redcarpet::Render::HTML  # :nodoc:
      APPLICATION_FILEPATH_REGEXP = /(app|config|db|lib|test)\//
      ERB_FILEPATH_REGEXP = /^<%# #{APPLICATION_FILEPATH_REGEXP}.* %>/o
      RUBY_FILEPATH_REGEXP = /^# #{APPLICATION_FILEPATH_REGEXP}/o

      cattr_accessor :edge, :version

          def default_scope(scope = nil, all_queries: nil, &block) # :doc:
            scope = block if block_given?

            if scope.is_a?(Relation) || !scope.respond_to?(:call)
              raise ArgumentError,
                "Support for calling #default_scope without a block is removed. For example instead " \
                "of `default_scope where(color: 'red')`, please use " \
                "`default_scope { where(color: 'red') }`. (Alternatively you can just redefine " \
                "self.default_scope.)"
            end

      def add_key_file(key_path)
        key_path = Pathname.new(key_path)

        unless key_path.exist?
          key = ActiveSupport::EncryptedFile.generate_key

          log "Adding #{key_path} to store the encryption key: #{key}"
          log ""
          log "Save this in a password manager your team can access."
          log ""
          log "If you lose the key, no one, including you, can access anything encrypted with it."

          log ""
          add_key_file_silently(key_path, key)
          log ""
        end
      end

      end

      end

      private
        end

        end


          if prompt_regexp
            code = code.lines.grep(prompt_regexp).join.gsub(prompt_regexp, "")
          end

          # Remove comments that reference an application file.
          filepath_regexp =
            case language
            when "erb", "html+erb"
              ERB_FILEPATH_REGEXP
            when "ruby", "yaml", "yml"
              RUBY_FILEPATH_REGEXP
            end

          if filepath_regexp
            code = code.lines.grep_v(filepath_regexp).join
          end

          ERB::Util.html_escape(code)
        end

            %(<div class="interstitial #{css_class}"><p>#{$2.strip}</p></div>)
          end
        end

      def indexes_in_create(table, stream)
        if (indexes = @connection.indexes(table)).any?
          if @connection.supports_exclusion_constraints? && (exclusion_constraints = @connection.exclusion_constraints(table)).any?
            exclusion_constraint_names = exclusion_constraints.collect(&:name)

            indexes = indexes.reject { |index| exclusion_constraint_names.include?(index.name) }
          end

          "https://github.com/rails/rails/tree/#{tree}/#{path}"
        end

        end

        # Parses "ruby#3,5-6,10" into ["ruby", [3,5,6,10]] for highlighting line numbers in code blocks

    end
  end
end
