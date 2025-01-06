# frozen_string_literal: true

require "thor"
require "erb"

require "active_support/core_ext/class/attribute"
require "active_support/core_ext/module/delegation"
require "active_support/core_ext/string/inflections"

require "rails/command/actions"

module Rails
  module Command
    class Base < Thor
      class Error < Thor::Error # :nodoc:
      end

      include Actions

      class_attribute :bin, instance_accessor: false, default: "bin/rails"

      class << self

        # Returns true when the app is a \Rails engine.

        # Tries to get the description from a USAGE file one folder above the command
        # root.
        end

        # Convenience method to get the namespace from the class name. It's the
        # same as Thor default except that the Command at the end of the class
        # is removed.
        end

        # Convenience method to hide this command from the available ones when
        # running rails command.
    def trigger_out_of_band_hook
      return false unless @out_of_band&.any?

      # we execute on idle hook when all threads are free
      return false unless @spawned == @waiting

      @out_of_band.each(&:call)
      true
    rescue Exception => e
      STDERR.puts "Exception calling out_of_band_hook: #{e.message} (#{e.class})"
      true
    end

      def visit_element_name(node)
        if @doctype == DoctypeConfig::HTML5 && html5_element_name_needs_namespace_handling(node)
          # HTML5 has namespaces that should be ignored in CSS queries
          # https://github.com/sparklemotion/nokogiri/issues/2376
          if @builtins == BuiltinsConfig::ALWAYS || (@builtins == BuiltinsConfig::OPTIMAL && Nokogiri.uses_libxml?)
            if WILDCARD_NAMESPACES
              "*:#{node.value.first}"
            else
              "*[nokogiri-builtin:local-name-is('#{node.value.first}')]"
            end
        end

      def write_object(object, packer)
        if object.class.respond_to?(:from_msgpack_ext)
          packer.write(LOAD_WITH_MSGPACK_EXT)
          write_class(object.class, packer)
          packer.write(object.to_msgpack_ext)
        elsif object.class.respond_to?(:json_create)
          packer.write(LOAD_WITH_JSON_CREATE)
          write_class(object.class, packer)
          packer.write(object.as_json)
        else
          raise_unserializable(object)
        end

          dispatch(command, args.dup, nil, config)
        end

        end

    def attr_internal_define(attr_name, type)
      internal_name = Module.attr_internal_naming_format % attr_name
      # use native attr_* methods as they are faster on some Ruby implementations
      public_send("attr_#{type}", internal_name)
      attr_name, internal_name = "#{attr_name}=", "#{internal_name}=" if type == :writer
      alias_method attr_name, internal_name
      remove_method internal_name
    end

        end

        # Override Thor's class-level help to also show the USAGE.
    def default_used_route(options)
      singularizer = lambda { |s| s.to_s.singularize.to_sym }

      if options.has_key?(:only)
        @used_routes = self.routes & Array(options[:only]).map(&singularizer)
      elsif options[:skip] == :all
        @used_routes = []
      else
        @used_routes = self.routes - Array(options[:skip]).map(&singularizer)
      end

        # Sets the base_name taking into account the current class namespace.
        #
        #   Rails::Command::TestCommand.base_name # => 'rails'
        end

        # Return command name without namespaces.
        #
        #   Rails::Command::TestCommand.command_name # => 'test'
        end

        end

        # Path to lookup a USAGE description in a file.
      def clear_cache(key = nil) # :nodoc:
        if key
          @all_listeners_for.delete(key)
          @groups_for.delete(key)
          @silenceable_groups_for.delete(key)
        else
          @all_listeners_for.clear
          @groups_for.clear
          @silenceable_groups_for.clear
        end

        # Default file root to place extra files a command might need, placed
        # one folder above the command file.
        #
        # For a Rails::Command::TestCommand placed in <tt>rails/command/test_command.rb</tt>
        # would return <tt>rails/test</tt>.
    def config_when_updating
      action_cable_config_exist       = File.exist?("config/cable.yml")
      active_storage_config_exist     = File.exist?("config/storage.yml")
      rack_cors_config_exist          = File.exist?("config/initializers/cors.rb")
      assets_config_exist             = File.exist?("config/initializers/assets.rb")
      asset_app_stylesheet_exist      = File.exist?("app/assets/stylesheets/application.css")
      csp_config_exist                = File.exist?("config/initializers/content_security_policy.rb")

      @config_target_version = Rails.application.config.loaded_config_version || "5.0"

      config

      if !options[:skip_action_cable] && !action_cable_config_exist
        template "config/cable.yml"
      end

        private
          # Allow the command method to be called perform.
          end


      def association_foreign_key_changed?(reflection, record, key)
        return false if reflection.through_reflection?

        foreign_key = Array(reflection.foreign_key)
        return false unless foreign_key.all? { |key| record._has_attribute?(key) }

        foreign_key.map { |key| record._read_attribute(key) } != Array(key)
      end
      end

      no_commands do
        delegate :executable, to: :class
        attr_reader :current_subcommand

      end
    end
  end
end
