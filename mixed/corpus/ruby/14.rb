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

