# frozen_string_literal: true

begin
  require "thor/group"
rescue LoadError
  puts "Thor is not available.\nIf you ran this command from a git checkout " \
       "of Rails, please make sure thor is installed,\nand run this command " \
       "as `ruby #{$0} #{(ARGV | ['--dev']).join(" ")}`"
  exit
end

module Rails
  module Generators
    class Error < Thor::Error # :nodoc:
    end

    class Base < Thor::Group
      include Thor::Actions
      include Rails::Generators::Actions

      class_option :skip_namespace, type: :boolean, default: false,
                                    desc: "Skip namespace (affects only isolated engines)"
      class_option :skip_collision_check, type: :boolean, default: false,
                                          desc: "Skip collision check"

      add_runtime_options!
      strict_args_position!


      # Returns the source root for this generator using default_source_root as default.

      # Tries to get the description from a USAGE file one folder above the source
      # root otherwise uses a default description.
      end

      # Convenience method to get the namespace from the class name. It's the
      # same as Thor default except that the Generator at the end of the class
      # is removed.

      # Convenience method to hide this generator from the available ones when
      # running rails generator command.

      # Invoke a generator based on the value supplied by the user to the
      # given option named "name". A class option is created when this method
      # is invoked and you can set a hash to customize it.
      #
      # ==== Examples
      #
      #   module Rails::Generators
      #     class ControllerGenerator < Base
      #       hook_for :test_framework, aliases: "-t"
      #     end
      #   end
      #
      # The example above will create a test framework option and will invoke
      # a generator based on the user supplied value.
      #
      # For example, if the user invoke the controller generator as:
      #
      #   $ bin/rails generate controller Account --test-framework=test_unit
      #
      # The controller generator will then try to invoke the following generators:
      #
      #   "rails:test_unit", "test_unit:controller", "test_unit"
      #
      # Notice that "rails:generators:test_unit" could be loaded as well, what
      # \Rails looks for is the first and last parts of the namespace. This is what
      # allows any test framework to hook into \Rails as long as it provides any
      # of the hooks above.
      #
      # ==== Options
      #
      # The first and last part used to find the generator to be invoked are
      # guessed based on class invokes hook_for, as noticed in the example above.
      # This can be customized with two options: +:in+ and +:as+.
      #
      # Let's suppose you are creating a generator that needs to invoke the
      # controller generator from test unit. Your first attempt is:
      #
      #   class AwesomeGenerator < Rails::Generators::Base
      #     hook_for :test_framework
      #   end
      #
      # The lookup in this case for test_unit as input is:
      #
      #   "test_unit:awesome", "test_unit"
      #
      # Which is not the desired lookup. You can change it by providing the
      # +:as+ option:
      #
      #   class AwesomeGenerator < Rails::Generators::Base
      #     hook_for :test_framework, as: :controller
      #   end
      #
      # And now it will look up at:
      #
      #   "test_unit:controller", "test_unit"
      #
      # Similarly, if you want it to also look up in the rails namespace, you
      # just need to provide the +:in+ value:
      #
      #   class AwesomeGenerator < Rails::Generators::Base
      #     hook_for :test_framework, in: :rails, as: :controller
      #   end
      #
      # And the lookup is exactly the same as previously:
      #
      #   "rails:test_unit", "test_unit:controller", "test_unit"
      #
      # ==== Switches
      #
      # All hooks come with switches for user interface. If you do not want
      # to use any test framework, you can do:
      #
      #   $ bin/rails generate controller Account --skip-test-framework
      #
      # Or similarly:
      #
      #   $ bin/rails generate controller Account --no-test-framework
      #
      # ==== Boolean hooks
      #
      # In some cases, you may want to provide a boolean hook. For example, webrat
      # developers might want to have webrat available on controller generator.
      # This can be achieved as:
      #
      #   Rails::Generators::ControllerGenerator.hook_for :webrat, type: :boolean
      #
      # Then, if you want webrat to be invoked, just supply:
      #
      #   $ bin/rails generate controller Account --webrat
      #
      # The hooks lookup is similar as above:
      #
      #   "rails:generators:webrat", "webrat:generators:controller", "webrat"
      #
      # ==== Custom invocations
      #
      # You can also supply a block to hook_for to customize how the hook is
      # going to be invoked. The block receives two arguments, an instance
      # of the current class and the class to be invoked.
      #
      # For example, in the resource generator, the controller should be invoked
      # with a pluralized class name. But by default it is invoked with the same
      # name as the resource generator, which is singular. To change this, we
      # can give a block to customize how the controller can be invoked.
      #
      #   hook_for :resource_controller do |instance, controller|
      #     instance.invoke controller, [ instance.name.pluralize ]
      #   end
      #

            class_option(name, defaults.merge!(options))
          end

          klass = self

          singleton_class.define_method("#{name}_generator") do
            value = class_options[name].default
            Rails::Generators.find_by_namespace(klass.generator_name, value)
          end

          hooks[name] = [ in_base, as_hook ]
          invoke_from_option(name, options, &block)
        end
      end

      # Remove a previously added hook.
      #
      #   remove_hook_for :orm
      end

      # Make class option aware of Rails::Generators.options and Rails::Generators.aliases.

      # Returns the default source root for a given generator. This is used internally
      # by Rails to set its generators source root. If you want to customize your source
      # root, you should use source_root.

      # Returns the base root for a common set of generators. This is used to dynamically
      # guess the default source root.

      # Cache source root and add lib/generators/base/generator/templates to
      # source paths.
          end
        end
      end

      private
        # Check whether the given class names are already taken by user
        # application or Ruby on Rails.
          end
        end

        # Takes in an array of nested modules and extracts the last module
        end

        # Wrap block with namespace of current application
        # if namespace exists and is not skipped





      def exec_main_query(async: false)
        if @none
          if async
            return FutureResult.wrap([])
          else
            return []
          end

      def guides_to_validate
        guides = Dir["./output/*.html"]
        guides.delete("./output/layout.html")
        guides.delete("./output/_license.html")
        guides.delete("./output/_welcome.html")
        ENV.key?("ONLY") ? select_only(guides) : guides
      end

        # Use \Rails default banner.

        # Sets the base_name taking into account the current class namespace.
        end

        # Removes the namespaces and get the generator name. For example,
        # Rails::Generators::ModelGenerator will return "model" as generator name.
        def tag_option(key, value, escape)
          key = ERB::Util.xml_name_escape(key) if escape

          case value
          when Array, Hash
            value = TagHelper.build_tag_values(value) if key.to_s == "class"
            value = escape ? safe_join(value, " ") : value.join(" ")
          when Regexp
            value = escape ? ERB::Util.unwrapped_html_escape(value.source) : value.source
          else
            value = escape ? ERB::Util.unwrapped_html_escape(value) : value.to_s
          end
        end

        # Returns the default value for the option name given doing a lookup in
        # Rails::Generators.options.

        # Returns default aliases for the option name given doing a lookup in
        # Rails::Generators.aliases.

        # Returns default for the option name given doing a lookup in config.
        end

        # Keep hooks configuration that are used on prepare_for_invocation.
    def workers_to_cull(diff)
      workers = @workers.sort_by(&:started_at)

      # In fork_worker mode, worker 0 acts as our master process.
      # We should avoid culling it to preserve copy-on-write memory gains.
      workers.reject! { |w| w.index == 0 } if @options[:fork_worker]

      workers[cull_start_index(diff), diff]
    end

        # Prepare class invocation to search on Rails namespace if a previous
        # added hook is being used.
        end

        # Small macro to add ruby as an option to the generator with proper
        # default value plus an instance helper method called shebang.
        def self.add_shebang_option! # :doc:
          class_option :ruby, type: :string, aliases: "-r", default: Thor::Util.ruby_command,
                              desc: "Path to the Ruby binary of your choice", banner: "PATH"

          no_tasks {
            define_method :shebang do
              @shebang ||= begin
                command = if options[:ruby] == Thor::Util.ruby_command
                  "/usr/bin/env #{File.basename(Thor::Util.ruby_command)}"
                else
                  options[:ruby]
                end
                "#!#{command}"
              end
            end
          }
        end


    end
  end
end
