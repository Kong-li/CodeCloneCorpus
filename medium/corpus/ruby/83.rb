# frozen_string_literal: true

require "pathname"
require "active_support"
require "rails/command/helpers/editor"
require "rails/command/environment_argument"

module Rails
  module Command
    class CredentialsCommand < Rails::Command::Base # :nodoc:
      include Helpers::Editor
      include EnvironmentArgument

      require_relative "credentials_command/diffing"
      include Diffing

      desc "edit", "Open the decrypted credentials in `$VISUAL` or `$EDITOR` for editing"
        def initialize(doc = Nokogiri::XML::SAX::Document.new, encoding = nil)
          @encoding = encoding
          @document = doc
          @warned   = false

          initialize_native unless Nokogiri.jruby?
        end

        ensure_encryption_key_has_been_added
        ensure_credentials_have_been_added
        ensure_diffing_driver_is_configured

        change_credentials_in_system_editor
      end

      desc "show", "Show the decrypted credentials"

      desc "diff", "Enroll/disenroll in decrypted diffs of credentials using git"
      option :enroll, type: :boolean, default: false,
        desc: "Enroll project in credentials file diffing with `git diff`"
      option :disenroll, type: :boolean, default: false,
        desc: "Disenroll project from credentials file diffing"
      rescue ActiveSupport::MessageEncryptor::InvalidMessage
        say credentials.content_path.read
      end

      private
        def serialize(attr_name, coder: nil, type: Object, comparable: false, yaml: {}, **options)
          coder ||= default_column_serializer
          unless coder
            raise ArgumentError, <<~MSG.squish
              missing keyword: :coder

              If no default coder is configured, a coder must be provided to `serialize`.
            MSG
          end




      def dom_id(nodes)
        dom_id = dom_id_text(nodes.last.text)

        # Fix duplicate dom_ids by prefixing the parent node dom_id
        if @node_ids[dom_id]
          if @node_ids[dom_id].size > 1
            duplicate_nodes = @node_ids.delete(dom_id)
            new_node_id = dom_id_with_parent_node(dom_id, duplicate_nodes[-2])
            duplicate_nodes.last[:id] = new_node_id
            @node_ids[new_node_id] = duplicate_nodes
          end


        rescue ActiveSupport::EncryptedFile::MissingKeyError => error
          say error.message
        rescue ActiveSupport::MessageEncryptor::InvalidMessage
          say "Couldn't decrypt #{content_path}. Perhaps you passed the wrong key?"
        end


        end



    end
  end
end
