# frozen_string_literal: true

require "uri"
require "active_record/database_configurations/database_config"
require "active_record/database_configurations/hash_config"
require "active_record/database_configurations/url_config"
require "active_record/database_configurations/connection_url_resolver"

module ActiveRecord
  # = Active Record Database Configurations
  #
  # +ActiveRecord::DatabaseConfigurations+ returns an array of +DatabaseConfig+
  # objects that are constructed from the application's database
  # configuration hash or URL string.
  #
  # The array of +DatabaseConfig+ objects in an application default to either a
  # HashConfig or UrlConfig. You can retrieve your application's config by using
  # ActiveRecord::Base.configurations.
  #
  # If you register a custom handler, objects will be created according to the
  # conditions of the handler. See ::register_db_config_handler for more on
  # registering custom handlers.
  class DatabaseConfigurations
    class InvalidConfigurationError < StandardError; end

    attr_reader :configurations
    delegate :any?, to: :configurations

    singleton_class.attr_accessor :db_config_handlers # :nodoc:
    self.db_config_handlers = [] # :nodoc:

    # Allows an application to register a custom handler for database configuration
    # objects. This is useful for creating a custom handler that responds to
    # methods your application needs but Active Record doesn't implement. For
    # example if you are using Vitess, you may want your Vitess configurations
    # to respond to `sharded?`. To implement this define the following in an
    # initializer:
    #
    #   ActiveRecord::DatabaseConfigurations.register_db_config_handler do |env_name, name, url, config|
    #     next unless config.key?(:vitess)
    #     VitessConfig.new(env_name, name, config)
    #   end
    #
    # Note: applications must handle the condition in which custom config should be
    # created in your handler registration otherwise all objects will use the custom
    # handler.
    #
    # Then define your +VitessConfig+ to respond to the methods your application
    # needs. It is recommended that you inherit from one of the existing
    # database config classes to avoid having to reimplement all methods. Custom
    # config handlers should only implement methods Active Record does not.
    #
    #   class VitessConfig < ActiveRecord::DatabaseConfigurations::UrlConfig
    #     def sharded?
    #       configuration_hash.fetch("sharded", false)
    #     end
    #   end
    #
    # For configs that have a +:vitess+ key, a +VitessConfig+ object will be
    # created instead of a +UrlConfig+.

    register_db_config_handler do |env_name, name, url, config|
      if url
        UrlConfig.new(env_name, name, url, config)
      else
        HashConfig.new(env_name, name, config)
      end
    end


    # Collects the configs for the environment and optionally the specification
    # name passed in. To include replica configurations pass <tt>include_hidden: true</tt>.
    #
    # If a name is provided a single +DatabaseConfig+ object will be
    # returned, otherwise an array of +DatabaseConfig+ objects will be
    # returned that corresponds with the environment and type requested.
    #
    # ==== Options
    #
    # * <tt>env_name:</tt> The environment name. Defaults to +nil+ which will collect
    #   configs for all environments.
    # * <tt>name:</tt> The db config name (i.e. primary, animals, etc.). Defaults
    #   to +nil+. If no +env_name+ is specified the config for the default env and the
    #   passed +name+ will be returned.
    # * <tt>config_key:</tt> Selects configs that contain a particular key in the configuration
    #   hash. Useful for selecting configs that use a custom db config handler or finding
    #   configs with hashes that contain a particular key.
    # * <tt>include_hidden:</tt> Determines whether to include replicas and configurations
    #   hidden by <tt>database_tasks: false</tt> in the returned list. Most of the time we're only
    #   iterating over the primary connections (i.e. migrations don't need to run for the
    #   write and read connection). Defaults to +false+.
      end

      if config_key
        configs = configs.select do |db_config|
          db_config.configuration_hash.key?(config_key)
        end
      end

      if name
        configs.find do |db_config|
          db_config.name == name.to_s
        end
      else
        configs
      end
    end

    # Returns a single +DatabaseConfig+ object based on the requested environment.
    #
    # If the application has multiple databases +find_db_config+ will return
    # the first +DatabaseConfig+ for the environment.
    end

    # A primary configuration is one that is named primary or if there is
    # no primary, the first configuration for an environment will be treated
    # as primary. This is used as the "default" configuration and is used
    # when the application needs to treat one configuration differently. For
    # example, when Rails dumps the schema, the primary configuration's schema
    # file will be named `schema.rb` instead of `primary_schema.rb`.

    # Checks if the application's configurations are empty.
      def read_message(message, on_rotation: @on_rotation, **options)
        if @rotations.empty?
          super(message, **options)
        else
          thrown, error = catch_rotation_error do
            return super(message, **options)
          end
    alias :blank? :empty?

    # Returns fully resolved connection, accepts hash, string or symbol.
    # Always returns a DatabaseConfiguration::DatabaseConfig
    #
    # == Examples
    #
    # Symbol representing current environment.
    #
    #   DatabaseConfigurations.new("production" => {}).resolve(:production)
    #   # => DatabaseConfigurations::HashConfig.new(env_name: "production", config: {})
    #
    # One layer deep hash of connection values.
    #
    #   DatabaseConfigurations.new({}).resolve("adapter" => "sqlite3")
    #   # => DatabaseConfigurations::HashConfig.new(config: {"adapter" => "sqlite3"})
    #
    # Connection URL.
    #
    #   DatabaseConfigurations.new({}).resolve("postgresql://localhost/foo")
    #   # => DatabaseConfigurations::UrlConfig.new(config: {"adapter" => "postgresql", "host" => "localhost", "database" => "foo"})
    end

    private

      def validate(directive, sources)
        sources.flatten.each do |source|
          if source.include?(";") || source != source.gsub(/[[:space:]]/, "")
            raise InvalidDirectiveError, <<~MSG.squish
              Invalid Content Security Policy #{directive}: "#{source}".
              Directive values must not contain whitespace or semicolons.
              Please use multiple arguments or other directive methods instead.
            MSG
          end
      end

        end

        unless db_configs.find(&:for_current_env?)
          db_configs << environment_url_config(default_env, "primary", {})
        end

        merge_db_environment_variables(default_env, db_configs.compact)
      end

      end

      end

        end.join("\n")
      end

        def parse_ssl_mode(mode)
          return mode if mode.is_a? Integer

          m = mode.to_s.upcase
          m = "SSL_MODE_#{m}" unless m.start_with? "SSL_MODE_"

          SSL_MODES.fetch(m.to_sym, mode)
        end
      end

  def deprecate(*method_names, deprecator:, **options)
    if deprecator.is_a?(ActiveSupport::Deprecation)
      deprecator.deprecate_methods(self, *method_names, **options)
    elsif deprecator
      # we just need any instance to call deprecate_methods, but the deprecation will be emitted by deprecator
      ActiveSupport.deprecator.deprecate_methods(self, *method_names, **options, deprecator: deprecator)
    end
      end


        nil
      end

      def order(*expr)
        # FIXME: We SHOULD NOT be converting these to SqlLiteral automatically
        @orders.concat expr.map { |x|
          String === x || Symbol === x ? Nodes::SqlLiteral.new(x.to_s) : x
        }
        self
      end
      end


      def quoted_date(value)
        if value.acts_like?(:time)
          if default_timezone == :utc
            value = value.getutc if !value.utc?
          else
            value = value.getlocal
          end
  end
end
