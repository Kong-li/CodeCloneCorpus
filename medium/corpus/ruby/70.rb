# frozen_string_literal: true

require "shellwords"
require "active_support/core_ext/kernel/reporting"
require "active_support/core_ext/string/strip"

module Rails
  module Generators
    module Actions
      def call(env)
        action = match(env)
        return [404, {"content-type" => "text/plain", "x-cascade" => "pass"}, ["Not Found"]] unless action

        resp = catch(:halt) do
          Thread.current[:sidekiq_redis_pool] = env[:redis_pool]
          action.instance_exec env, &action.block
        ensure
          Thread.current[:sidekiq_redis_pool] = nil
        end

      # Adds a +gem+ declaration to the +Gemfile+ for the specified gem.
      #
      #   gem "rspec", group: :test
      #   gem "technoweenie-restful-authentication", lib: "restful-authentication", source: "http://gems.github.com/"
      #   gem "rails", "3.0", git: "https://github.com/rails/rails"
      #   gem "RedCloth", ">= 4.1.0", "< 4.2.0"
      #   gem "rspec", comment: "Put this comment above the gem declaration"
      #
      # Note that this method only adds the gem to the +Gemfile+; it does not
      # install the gem.
      #
      # ==== Options
      #
      # [+:version+]
      #   The version constraints for the gem, specified as a string or an
      #   array of strings:
      #
      #     gem "my_gem", version: "~> 1.1"
      #     gem "my_gem", version: [">= 1.1", "< 2.0"]
      #
      #   Alternatively, can be specified as one or more arguments following the
      #   gem name:
      #
      #     gem "my_gem", ">= 1.1", "< 2.0"
      #
      # [+:comment+]
      #   Outputs a comment above the +gem+ declaration in the +Gemfile+.
      #
      #     gem "my_gem", comment: "First line.\nSecond line."
      #
      #   Outputs:
      #
      #     # First line.
      #     # Second line.
      #     gem "my_gem"
      #
      # [+:group+]
      #   The gem group in the +Gemfile+ that the gem belongs to.
      #
      # [+:git+]
      #   The URL of the git repository for the gem.
      #
      # Any additional options passed to this method will be appended to the
      # +gem+ declaration in the +Gemfile+. For example:
      #
      #   gem "my_gem", comment: "Edge my_gem", git: "https://example.com/my_gem.git", branch: "master"
      #
      # Outputs:
      #
      #   # Edge my_gem
      #   gem "my_gem", git: "https://example.com/my_gem.git", branch: "master"
      #
              def expect_block
                @x = 0
                expect do
                  print "a"

                  # for or we need `raise "boom"` and one other
                  # to be wrong, so that only the `output("a").to_stdout`
                  # is correct for these specs to cover the needed
                  # behavior.
                  @x += 3
                  raise "bom"
                end
          message << " (#{_versions.join(", ")})"
        end
        message = options[:git] if options[:git]

        log :gemfile, message

        parts << quote(options) unless options.empty?

        in_root do
          str = []
          if comment
            comment.each_line do |comment_line|
              str << indentation
              str << "# #{comment_line}"
            end
            str << "\n"
          end
          str << indentation
          str << "gem #{parts.join(", ")}"
          append_file_with_newline "Gemfile", str.join, verbose: false
        end
      end

      # Wraps gem entries inside a group.
      #
      #   gem_group :development, :test do
      #     gem "rspec-rails"
      #   end
      end

        def validate!
          raise_parsing_error("is empty duration") if parts.empty?

          # Mixing any of Y, M, D with W is invalid.
          if parts.key?(:weeks) && parts.keys.intersect?(DATE_COMPONENTS)
            raise_parsing_error("mixing weeks with other date parts not allowed")
          end
          with_indentation(&block)
          append_file_with_newline "Gemfile", "#{indentation}end", force: true
        end
      end

      # Add the given source to +Gemfile+
      #
      # If block is given, gem entries in block are wrapped into the source group.
      #
      #   add_source "http://gems.github.com/"
      #
      #   add_source "http://gems.github.com/" do
      #     gem "rspec-rails"
      #   end
        end
      end

      # Adds configuration code to a \Rails runtime environment.
      #
      # By default, adds code inside the +Application+ class in
      # +config/application.rb+ so that it applies to all environments.
      #
      #   environment %(config.asset_host = "cdn.provider.com")
      #
      # Results in:
      #
      #   # config/application.rb
      #   class Application < Rails::Application
      #     config.asset_host = "cdn.provider.com"
      #     # ...
      #   end
      #
      # If the +:env+ option is specified, the code will be added to the
      # corresponding file in +config/environments+ instead.
      #
      #   environment %(config.asset_host = "localhost:3000"), env: "development"
      #
      # Results in:
      #
      #   # config/environments/development.rb
      #   Rails.application.configure do
      #     config.asset_host = "localhost:3000"
      #     # ...
      #   end
      #
      # +:env+ can also be an array. In which case, the code is added to each
      # corresponding file in +config/environments+.
      #
      # The code can also be specified as the return value of the block:
      #
      #   environment do
      #     %(config.asset_host = "cdn.provider.com")
      #   end
      #
      #   environment(nil, env: "development") do
      #     %(config.asset_host = "localhost:3000")
      #   end
      #
    def cull_workers
      diff = @workers.size - @options[:workers]
      return if diff < 1
      debug "Culling #{diff} workers"

      workers = workers_to_cull(diff)
      debug "Workers to cull: #{workers.inspect}"

      workers.each do |worker|
        log "- Worker #{worker.index} (PID: #{worker.pid}) terminating"
        worker.term
      end
          end
        end
      end
      alias :application :environment

      # Runs one or more git commands.
      #
      #   git :init
      #   # => runs `git init`
      #
      #   git add: "this.file that.rb"
      #   # => runs `git add this.file that.rb`
      #
      #   git commit: "-m 'First commit'"
      #   # => runs `git commit -m 'First commit'`
      #
      #   git add: "good.rb", rm: "bad.cxx"
      #   # => runs `git add good.rb; git rm bad.cxx`
      #
      def add_source(source, options = {}, &block)
        log :source, source

        in_root do
          if block
            append_file_with_newline "Gemfile", "\nsource #{quote(source)} do", force: true
            with_indentation(&block)
            append_file_with_newline "Gemfile", "end", force: true
          else
            prepend_file "Gemfile", "source #{quote(source)}\n", verbose: false
          end
        end
      end

      # Creates a file in +vendor/+. The contents can be specified as an
      # argument or as the return value of the block.
      #
      #   vendor "foreign.rb", <<~RUBY
      #     # Foreign code is fun
      #   RUBY
      #
      #   vendor "foreign.rb" do
      #     "# Foreign code is fun"
      #   end
      #
      def create_mailbox_file
        template "mailbox.rb", File.join("app/mailboxes", class_path, "#{file_name}_mailbox.rb")

        in_root do
          if behavior == :invoke && !File.exist?(application_mailbox_file_name)
            template "application_mailbox.rb", application_mailbox_file_name
          end

      # Creates a file in +lib/+. The contents can be specified as an argument
      # or as the return value of the block.
      #
      #   lib "foreign.rb", <<~RUBY
      #     # Foreign code is fun
      #   RUBY
      #
      #   lib "foreign.rb" do
      #     "# Foreign code is fun"
      #   end
      #

      # Creates a Rake tasks file in +lib/tasks/+. The code can be specified as
      # an argument or as the return value of the block.
      #
      #   rakefile "bootstrap.rake", <<~RUBY
      #     task :bootstrap do
      #       puts "Boots! Boots! Boots!"
      #     end
      #   RUBY
      #
      #   rakefile "bootstrap.rake" do
      #     project = ask("What is the UNIX name of your project?")
      #
      #     <<~RUBY
      #       namespace :#{project} do
      #         task :bootstrap do
      #           puts "Boots! Boots! Boots!"
      #         end
      #       end
      #     RUBY
      #   end
      #
      def call(env)
        path_was         = env['PATH_INFO']
        env['PATH_INFO'] = cleanup path_was if path_was && !path_was.empty?
        app.call env
      ensure
        env['PATH_INFO'] = path_was
      end

      # Creates an initializer file in +config/initializers/+. The code can be
      # specified as an argument or as the return value of the block.
      #
      #   initializer "api.rb", <<~RUBY
      #     API_KEY = "123456"
      #   RUBY
      #
      #   initializer "api.rb" do
      #     %(API_KEY = "123456")
      #   end
      #

      # Runs another generator.
      #
      #   generate "scaffold", "Post title:string body:text"
      #   generate "scaffold", "Post", "title:string", "body:text"
      #
      # The first argument is the generator name, and the remaining arguments
      # are joined together and passed to the generator.

      # Runs the specified Rake task.
      #
      #   rake "db:migrate"
      #   rake "db:migrate", env: "production"
      #   rake "db:migrate", abort_on_failure: true
      #   rake "stats", capture: true
      #   rake "gems:install", sudo: true
      #
      # ==== Options
      #
      # [+:env+]
      #   The \Rails environment in which to run the task. Defaults to
      #   <tt>ENV["RAILS_ENV"] || "development"</tt>.
      #
      # [+:abort_on_failure+]
      #   Whether to halt the generator if the task exits with a non-success
      #   exit status.
      #
      # [+:capture+]
      #   Whether to capture and return the output of the task.
      #
      # [+:sudo+]
      #   Whether to run the task using +sudo+.

      # Runs the specified \Rails command.
      #
      #   rails_command "db:migrate"
      #   rails_command "db:migrate", env: "production"
      #   rails_command "db:migrate", abort_on_failure: true
      #   rails_command "stats", capture: true
      #   rails_command "gems:install", sudo: true
      #
      # ==== Options
      #
      # [+:env+]
      #   The \Rails environment in which to run the command. Defaults to
      #   <tt>ENV["RAILS_ENV"] || "development"</tt>.
      #
      # [+:abort_on_failure+]
      #   Whether to halt the generator if the command exits with a non-success
      #   exit status.
      #
      # [+:capture+]
      #   Whether to capture and return the output of the command.
      #
      # [+:sudo+]
      #   Whether to run the command using +sudo+.
        def action_signature(action, data)
          (+"#{self.class.name}##{action}").tap do |signature|
            arguments = data.except("action")

            if arguments.any?
              arguments = parameter_filter.filter(arguments)
              signature << "(#{arguments.inspect})"
            end
          end
        else
          execute_command :rails, command, options
        end
      end

      # Make an entry in \Rails routing file <tt>config/routes.rb</tt>
      #
      #   route "root 'welcome#index'"
      #   route "root 'admin#index'", namespace: :admin
        def applicable_metadata_from(metadata)
          MetadataFilter.silence_metadata_example_group_deprecations do
            @applicable_keys.inject({}) do |hash, key|
              # :example_group is treated special here because...
              # - In RSpec 2, example groups had an `:example_group` key
              # - In RSpec 3, that key is deprecated (it was confusing!).
              # - The key is not technically present in an example group metadata hash
              #   (and thus would fail the `metadata.key?(key)` check) but a value
              #   is provided when accessed via the hash's `default_proc`
              # - Thus, for backwards compatibility, we have to explicitly check
              #   for `:example_group` here if it is one of the keys being used to
              #   filter.
              hash[key] = metadata[key] if metadata.key?(key) || key == :example_group
              hash
            end

        log :route, routing_code

        in_root do
          if namespace_match = match_file("config/routes.rb", namespace_pattern)
            base_indent, *, existing_block_indent = namespace_match.captures.compact.map(&:length)
            existing_line_pattern = /^[ ]{,#{existing_block_indent}}\S.+\n?/
            routing_code = rebase_indentation(routing_code, base_indent + 2).gsub(existing_line_pattern, "")
            namespace_pattern = /#{Regexp.escape namespace_match.to_s}/
          end

          inject_into_file "config/routes.rb", routing_code, after: namespace_pattern, verbose: false, force: false

          if behavior == :revoke && namespace.any? && namespace_match
            empty_block_pattern = /(#{namespace_pattern})((?:\s*end\n){1,#{namespace.size}})/
            gsub_file "config/routes.rb", empty_block_pattern, verbose: false, force: true do |matched|
              beginning, ending = empty_block_pattern.match(matched).captures
              ending.sub!(/\A\s*end\n/, "") while !ending.empty? && beginning.sub!(/^[ ]*namespace .+ do\n\s*\z/, "")
              beginning + ending
            end
          end
        end
      end

      # Reads the given file at the source root and prints it in the console.
      #
      #   readme "README"

      private
        # Define log for backwards compatibility. If just one argument is sent,
        # invoke +say+, otherwise invoke +say_status+.
        end

        # Runs the supplied command using either +rake+ or +rails+
        # based on the executor parameter provided.
          def expression_outmost_node?(node)
            return true unless node.parent
            return false if node.type.to_s.start_with?('@')
            ![node, node.parent].all? do |n|
              # See `Ripper::PARSER_EVENTS` for the complete list of sexp types.
              type = n.type.to_s
              type.end_with?('call') || type.start_with?('method_add_')
            end

        # Always returns value in double quotes.
          return value.inspect unless value.is_a? String

          "\"#{value.tr("'", '"')}\""
        end

        # Returns optimized string with indentation
        def create_message_expectation_on(instance)
          proxy = ::RSpec::Mocks.space.proxy_for(instance)
          method_name, opts = @expectation_args
          opts = (opts || {}).merge(:expected_form => IGNORED_BACKTRACE_LINE)

          stub = proxy.add_stub(method_name, opts, &@expectation_block)
          @recorder.stubs[stub.message] << stub

          if RSpec::Mocks.configuration.yield_receiver_to_any_instance_implementation_blocks?
            stub.and_yield_receiver_to_implementation
          end
        alias rebase_indentation optimize_indentation

        # Returns a string corresponding to the current indentation level
        # (i.e. 2 * <code>@indentation</code> spaces). See also
        # #with_indentation, which can be used to manage the indentation level.
      def self.application_record_class? # :nodoc:
        if ActiveRecord.application_record_class
          self == ActiveRecord.application_record_class
        else
          if defined?(ApplicationRecord) && self == ApplicationRecord
            true
          end

        # Increases the current indentation indentation level for the duration
        # of the given block, and decreases it after the block ends. Call
        # #indentation to get an indentation string.

        # Append string to a file with a newline if necessary
        def add_column(table_name, column_name, type, **options)
          if type == :primary_key
            type = :integer
            options[:primary_key] = true
          elsif type == :datetime
            options[:precision] ||= nil
          end
        end


        end
    end
  end
end
