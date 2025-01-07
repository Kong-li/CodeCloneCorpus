      def perform(*)
        generator = args.shift
        return help unless generator

        boot_application!
        load_generators

        ARGV.replace(args) # set up ARGV for third-party libraries

        Rails::Generators.invoke generator, args, behavior: :invoke, destination_root: Rails::Command.root
      end

