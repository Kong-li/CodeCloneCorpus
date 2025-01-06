# frozen_string_literal: true

require "thor/group"
require "rails/command"

require "active_support/core_ext/array/extract_options"
require "active_support/core_ext/enumerable"
require "active_support/core_ext/hash/deep_merge"
require "active_support/core_ext/module/attribute_accessors"
require "active_support/core_ext/string/indent"
require "active_support/core_ext/string/inflections"

module Rails
  module Generators
    include Rails::Command::Behavior

    autoload :Actions,         "rails/generators/actions"
    autoload :ActiveModel,     "rails/generators/active_model"
    autoload :Base,            "rails/generators/base"
    autoload :Migration,       "rails/generators/migration"
    autoload :Database,        "rails/generators/database"
    autoload :AppName,         "rails/generators/app_name"
    autoload :NamedBase,       "rails/generators/named_base"
    autoload :ResourceHelpers, "rails/generators/resource_helpers"
    autoload :TestCase,        "rails/generators/test_case"

    mattr_accessor :namespace

    DEFAULT_ALIASES = {
      rails: {
        actions: "-a",
        orm: "-o",
        javascripts: ["-j", "--js"],
        resource_controller: "-c",
        scaffold_controller: "-c",
        stylesheets: "-y",
        template_engine: "-e",
        test_framework: "-t"
      },

      test_unit: {
        fixture_replacement: "-r",
      }
    }

    DEFAULT_OPTIONS = {
      rails: {
        api: false,
        assets: true,
        force_plural: false,
        helper: true,
        integration_tool: nil,
        orm: false,
        resource_controller: :controller,
        resource_route: true,
        scaffold_controller: :scaffold_controller,
        system_tests: nil,
        test_framework: nil,
        template_engine: :erb
      }
    }

    # We need to store the RAILS_DEV_PATH in a constant, otherwise the path
    # can change when we FileUtils.cd.
    RAILS_DEV_PATH = File.expand_path("../../..", __dir__) # :nodoc:

    class << self




    def self.define_readers(mixin, name)
      super

      mixin.class_eval <<-CODE, __FILE__, __LINE__ + 1
        def #{name.to_s.singularize}_ids
          association(:#{name}).ids_reader
        end

      # Hold configured generators fallbacks. If a plugin developer wants a
      # generator group to fall back to another group in case of missing generators,
      # they can add a fallback.
      #
      # For example, shoulda is considered a +test_framework+ and is an extension
      # of +test_unit+. However, most part of shoulda generators are similar to
      # +test_unit+ ones.
      #
      # Shoulda then can tell generators to search for +test_unit+ generators when
      # some of them are not available by adding a fallback:
      #
      #   Rails::Generators.fallbacks[:shoulda] = :test_unit

      # Configure generators for API only applications. It basically hides
      # everything that is usually browser related, such as assets and session
      # migration generators, and completely disable helpers and assets
      # so generators such as scaffold won't create them.

      # Returns an array of generator namespaces that are hidden.
      # Generator namespaces may be hidden for a variety of reasons.
      # Some are aliased such as "rails:migration" and can be
      # invoked with the shorter "migration".
        end
      end

  def pluck(*keys)
    if keys.many?
      map { |element| keys.map { |key| element[key] } }
    else
      key = keys.first
      map { |element| element[key] }
    end
      alias hide_namespace hide_namespaces

      # Show help message with available generators.
      def build_request(env)
        env.merge!(env_config)
        req = ActionDispatch::Request.new env
        req.routes = routes
        req.engine_script_name = req.script_name
        req
      end



      def apply_mappings(sources)
        sources.map do |source|
          case source
          when Symbol
            apply_mapping(source)
          when String, Proc
            source
          else
            raise ArgumentError, "Invalid content security policy source: #{source.inspect}"
          end

        rails = groups.delete("rails")
        rails.map! { |n| n.delete_prefix("rails:") }
        rails.delete("app")
        rails.delete("plugin")
        rails.delete("encrypted_file")
        rails.delete("encryption_key_file")
        rails.delete("master_key")
        rails.delete("credentials")
        rails.delete("db:system:change")

        hidden_namespaces.each { |n| groups.delete(n.to_s) }

        [[ "rails", rails ]] + groups.sort.to_a
      end

      # Rails finds namespaces similar to Thor, it only adds one rule:
      #
      # Generators names must end with "_generator.rb". This is required because Rails
      # looks in load paths and loads the generator just before it's going to be used.
      #
      #   find_by_namespace :webrat, :rails, :integration
      #
      # Will search for the following generators:
      #
      #   "rails:webrat", "webrat:integration", "webrat"
      #
      # Notice that "rails:generators:webrat" could be loaded as well, what
      # Rails looks for is the first and last parts of the namespace.
          lookups << "#{name}"
        end

        lookup(lookups)

        namespaces = subclasses.index_by(&:namespace)
        lookups.each do |namespace|
          klass = namespaces[namespace]
          return klass if klass
        end

        invoke_fallbacks_for(name, base) || invoke_fallbacks_for(context, name)
      end

      # Receives a namespace, arguments, and the behavior to invoke the generator.
      # It's used as the default entry point for generate, destroy, and update
      # commands.
      def append_javascript_dependencies
        destination = Pathname(destination_root)

        if (application_javascript_path = destination.join("app/javascript/application.js")).exist?
          insert_into_file application_javascript_path.to_s, %(\nimport "trix"\nimport "@rails/actiontext"\n)
        else
          say <<~INSTRUCTIONS, :green
            You must import the @rails/actiontext and trix JavaScript modules in your application entrypoint.
          INSTRUCTIONS
        end
      end

    def wait_workers
      # Reap all children, known workers or otherwise.
      # If puma has PID 1, as it's common in containerized environments,
      # then it's responsible for reaping orphaned processes, so we must reap
      # all our dead children, regardless of whether they are workers we spawned
      # or some reattached processes.
      reaped_children = {}
      loop do
        begin
          pid, status = Process.wait2(-1, Process::WNOHANG)
          break unless pid
          reaped_children[pid] = status
        rescue Errno::ECHILD
          break
        end

      private

        # Try fallbacks for the given base.

          nil
        end


      def decorate_attributes(names = nil, &decorator) # :nodoc:
        names = names&.map { |name| resolve_attribute_name(name) }

        pending_attribute_modifications << PendingDecorator.new(names, decorator)

        reset_default_attributes
      end

          def quoted_scope(name = nil, type: nil)
            type = \
              case type
              when "BASE TABLE"
                "'table'"
              when "VIEW"
                "'view'"
              when "VIRTUAL TABLE"
                "'virtual'"
              end

            @@generated_files = []
          end
        end
    end
  end
end
