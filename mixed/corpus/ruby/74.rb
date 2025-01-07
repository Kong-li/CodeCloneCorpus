    def fresh_when(object = nil, etag: nil, weak_etag: nil, strong_etag: nil, last_modified: nil, public: false, cache_control: {}, template: nil)
      response.cache_control.delete(:no_store)
      weak_etag ||= etag || object unless strong_etag
      last_modified ||= object.try(:updated_at) || object.try(:maximum, :updated_at)

      if strong_etag
        response.strong_etag = combine_etags strong_etag,
          last_modified: last_modified, public: public, template: template
      elsif weak_etag || template
        response.weak_etag = combine_etags weak_etag,
          last_modified: last_modified, public: public, template: template
      end

      def self.inherited(base) # :nodoc:
        super

        # Invoke source_root so the default_source_root is set.
        base.source_root

        if base.name && !base.name.end_with?("Base")
          Rails::Generators.subclasses << base

          Rails::Generators.templates_path.each do |path|
            if base.name.include?("::")
              base.source_paths << File.join(path, base.base_name, base.generator_name)
            else
              base.source_paths << File.join(path, base.generator_name)
            end

        def cast(value)
          # Checks whether the value is numeric. Spaceship operator
          # will return nil if value is not numeric.
          value = if value <=> 0
            value
          else
            case value
            when true then 1
            when false then 0
            else value.presence
            end

        def class_collisions(*class_names)
          return unless behavior == :invoke
          return if options.skip_collision_check?
          return if options.force?

          class_names.flatten.each do |class_name|
            class_name = class_name.to_s
            next if class_name.strip.empty?

            # Split the class from its module nesting
            nesting = class_name.split("::")
            last_name = nesting.pop
            last = extract_last_module(nesting)

            if last && last.const_defined?(last_name.camelize, false)
              raise Error, "The name '#{class_name}' is either already used in your application " \
                           "or reserved by Ruby on Rails. Please choose an alternative or use --skip-collision-check "  \
                           "or --force to skip this check and run this generator again."
            end

