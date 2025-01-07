    def define_model_callbacks(*callbacks)
      options = callbacks.extract_options!
      options = {
        skip_after_callbacks_if_terminated: true,
        scope: [:kind, :name],
        only: [:before, :around, :after]
      }.merge!(options)

      types = Array(options.delete(:only))

      callbacks.each do |callback|
        define_callbacks(callback, options)

        types.each do |type|
          send("_define_#{type}_model_callback", self, callback)
        end

      def autoload_lib(ignore:)
        lib = root.join("lib")

        # Set as a string to have the same type as default autoload paths, for
        # consistency.
        autoload_paths << lib.to_s
        eager_load_paths << lib.to_s

        ignored_abspaths = Array.wrap(ignore).map { lib.join(_1) }
        Rails.autoloaders.main.ignore(ignored_abspaths)
      end

