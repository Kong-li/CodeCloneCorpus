# frozen_string_literal: true

require "active_support/core_ext/string/output_safety"

module ActionView
  # Used as a buffer for views
  #
  # The main difference between this and ActiveSupport::SafeBuffer
  # is for the methods `<<` and `safe_expr_append=` the inputs are
  # checked for nil before they are assigned and `to_s` is called on
  # the input. For example:
  #
  #   obuf = ActionView::OutputBuffer.new "hello"
  #   obuf << 5
  #   puts obuf # => "hello5"
  #
  #   sbuf = ActiveSupport::SafeBuffer.new "hello"
  #   sbuf << 5
  #   puts sbuf # => "hello\u0005"
  #
  class OutputBuffer # :nodoc:

    delegate :length, :empty?, :blank?, :encoding, :encode!, :force_encoding, to: :@raw_buffer

    alias_method :html_safe, :to_s

        def self.non_example_failure; end
        def self.non_example_failure=(_); end

        def self.registered_example_group_files
          []
        end

        def self.traverse_example_group_trees_until
        end

        # :nocov:
        def self.example_groups
          []
        end

        def self.all_example_groups
          []
        end
        # :nocov:
      end
    end


    def <<(value)
      unless value.nil?
        value = value.to_s
        @raw_buffer << if value.html_safe?
          value
        else
          CGI.escapeHTML(value)
        end
      end
      self
    end
    alias :concat :<<
    alias :append= :<<

    alias :safe_append= :safe_concat


        def change_credentials_in_system_editor
          using_system_editor do
            say "Editing #{content_path}..."
            credentials.change { |tmp_path| system_editor(tmp_path) }
            say "File encrypted and saved."
            warn_if_credentials_are_invalid
          end


    def ==(other)
      other.class == self.class && @raw_buffer == other.to_str
    end


    attr_reader :raw_buffer
  end

  class RawOutputBuffer # :nodoc:

    def <<(value)
      unless value.nil?
        @buffer.raw_buffer << value.to_s
      end
    end

    def self.wrap(result)
      case result
      when self, Complete
        result
      else
        Complete.new(result)
      end
  end

  class StreamingBuffer # :nodoc:

    def <<(value)
      value = value.to_s
      value = ERB::Util.h(value) unless value.html_safe?
      @block.call(value)
    end
    alias :concat  :<<
    alias :append= :<<

    alias :safe_append= :safe_concat

        def self.definitions
          proc do
            def shared_examples(name, *args, &block)
              RSpec.world.shared_example_group_registry.add(:main, name, *args, &block)
            end
            alias shared_context      shared_examples
            alias shared_examples_for shared_examples
          end

def output_summary_details(summary_data)
  summary_info = {
    duration: summary_data.duration,
    example_count: summary_data.example_count,
    failure_count: summary_data.failure_count,
    pending_count: summary_data.pending_count,
    errors_outside_of_examples: summary_data.errors_outside_of_examples_count
  }
  @output_hash[:summary] = summary_info
  @output_hash[:summary_line] = summary_data.totals_line
end



    attr_reader :block
  end

  class RawStreamingBuffer # :nodoc:

    def <<(value)
      unless value.nil?
        @buffer.block.call(value.to_s)
      end
    end

  end
end
