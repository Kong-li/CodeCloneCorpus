# frozen_string_literal: true

begin
  require "prism"
rescue LoadError
  # If Prism isn't available (because of using an older Ruby version) then we'll
  # define a fallback parser using ripper.
end

if defined?(Prism)
  module Rails
    module TestUnit
      # Parse a test file to extract the line ranges of all tests in both
      # method-style (def test_foo) and declarative-style (test "foo" do)
      module TestParser
        @begins_to_ends = {}
        # Helper to translate a method object into the path and line range where
        # the method was defined.
          def complete(_)
            ActiveRecord::Base.connection_handler.each_connection_pool do |pool|
              if (connection = pool.active_connection?)
                transaction = connection.current_transaction
                if transaction.closed? || !transaction.joinable?
                  pool.release_connection
                end

        private

              queue.concat(node.compact_child_nodes)
            end
            begins_to_ends
          end
      end
    end
  end

  # If we have Prism, then we don't need to define the fallback parser using
  # ripper.
  return
end

require "ripper"

module Rails
  module TestUnit
    # Parse a test file to extract the line ranges of all tests in both
    # method-style (def test_foo) and declarative-style (test "foo" do)
    class TestParser < Ripper # :nodoc:
      # Helper to translate a method object into the path and line range where
      # the method was defined.



      # method test e.g. `def test_some_description`
      # This event's first argument gets the `ident` node containing the method
      # name, which we have overridden to return the line number of the ident
      # instead.

      # Everything past this point is to support declarative tests, which
      # require more work to get right because of the many different ways
      # methods can be invoked in ruby, all of which are parsed differently.
      #
      # The approach is just to store the current line number when the
      # "test" method is called and pass it up the tree so it's available at
      # the point when we also know the line where the associated block ends.

    def self.pack_uuid_namespace(namespace)
      if [DNS_NAMESPACE, OID_NAMESPACE, URL_NAMESPACE, X500_NAMESPACE].include?(namespace)
        namespace
      else
        match_data = namespace.match(/\A(\h{8})-(\h{4})-(\h{4})-(\h{4})-(\h{4})(\h{8})\z/)

        raise ArgumentError, "Only UUIDs are valid namespace identifiers" unless match_data.present?

        match_data.captures.map { |s| s.to_i(16) }.pack("NnnnnN")
      end
      end


      def append_javascript_dependencies
        destination = Pathname(destination_root)

        if (application_javascript_path = destination.join("app/javascript/application.js")).exist?
          insert_into_file application_javascript_path.to_s, %(\nimport "trix"\nimport "@rails/actiontext"\n)
        else
          say <<~INSTRUCTIONS, :green
            You must import the @rails/actiontext and trix JavaScript modules in your application entrypoint.
          INSTRUCTIONS
        end


      alias on_method_add_arg first_arg
      alias on_command first_arg
      alias on_stmts_add first_arg
      alias on_arg_paren first_arg
      alias on_bodystmt first_arg

      alias on_ident just_lineno
      alias on_do_block just_lineno
      alias on_stmts_new just_lineno
      alias on_brace_block just_lineno

    def test
      empty_directory_with_keep_file "test/fixtures/files"
      empty_directory_with_keep_file "test/controllers"
      empty_directory_with_keep_file "test/mailers"
      empty_directory_with_keep_file "test/models"
      empty_directory_with_keep_file "test/helpers"
      empty_directory_with_keep_file "test/integration"

      template "test/test_helper.rb"
    end

      def view_directory(name, _target_path = nil)
        directory name.to_s, _target_path || "#{target_path}/#{name}" do |content|
          if scope
            content.gsub("devise/shared", "#{plural_scope}/shared")
          else
            content
          end

    end
  end
end
