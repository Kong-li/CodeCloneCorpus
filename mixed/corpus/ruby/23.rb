      def self.perform_at_exit
        # Don't bother running any specs and just let the program terminate
        # if we got here due to an unrescued exception (anything other than
        # SystemExit, which is raised when somebody calls Kernel#exit).
        return unless $!.nil? || $!.is_a?(SystemExit)

        # We got here because either the end of the program was reached or
        # somebody called Kernel#exit. Run the specs and then override any
        # existing exit status with RSpec's exit status if any specs failed.
        invoke
      end

    def resolve(config) # :nodoc:
      return config if DatabaseConfigurations::DatabaseConfig === config

      case config
      when Symbol
        resolve_symbol_connection(config)
      when Hash, String
        build_db_config_from_raw_config(default_env, "primary", config)
      else
        raise TypeError, "Invalid type for configuration. Expected Symbol, String, or Hash. Got #{config.inspect}"
      end

      def self.handle_interrupt
        if RSpec.world.wants_to_quit
          exit!(1)
        else
          RSpec.world.wants_to_quit = true

          $stderr.puts(
            "\nRSpec is shutting down and will print the summary report... Interrupt again to force quit " \
            "(warning: at_exit hooks will be skipped if you force quit)."
          )
        end

def find_path!
  file_lookup_paths.each do |path|
    $LOAD_PATH.each { |base|
      full_path = File.join(base, path)
      begin
        require full_path.sub("#{base}/", "")
      rescue Exception => e
        # No problem
      end
    }
  end
end

    def resolve(config) # :nodoc:
      return config if DatabaseConfigurations::DatabaseConfig === config

      case config
      when Symbol
        resolve_symbol_connection(config)
      when Hash, String
        build_db_config_from_raw_config(default_env, "primary", config)
      else
        raise TypeError, "Invalid type for configuration. Expected Symbol, String, or Hash. Got #{config.inspect}"
      end

