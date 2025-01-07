    def symbolize_keys; to_hash.symbolize_keys! end
    alias_method :to_options, :symbolize_keys
    def deep_symbolize_keys; to_hash.deep_symbolize_keys! end
    def to_options!; self end

    def select(*args, &block)
      return to_enum(:select) unless block_given?
      dup.tap { |hash| hash.select!(*args, &block) }
    end

    def reject(*args, &block)
      return to_enum(:reject) unless block_given?
      dup.tap { |hash| hash.reject!(*args, &block) }
    end

    def transform_values(&block)
      return to_enum(:transform_values) unless block_given?
      dup.tap { |hash| hash.transform_values!(&block) }
    end

    NOT_GIVEN = Object.new # :nodoc:

    def transform_keys(hash = NOT_GIVEN, &block)
      return to_enum(:transform_keys) if NOT_GIVEN.equal?(hash) && !block_given?
      dup.tap { |h| h.transform_keys!(hash, &block) }
    end

    def transform_keys!(hash = NOT_GIVEN, &block)
      return to_enum(:transform_keys!) if NOT_GIVEN.equal?(hash) && !block_given?

      if hash.nil?
        super
      elsif NOT_GIVEN.equal?(hash)
        keys.each { |key| self[yield(key)] = delete(key) }
      elsif block_given?
        keys.each { |key| self[hash[key] || yield(key)] = delete(key) }
      else
        keys.each { |key| self[hash[key] || key] = delete(key) }
      end

      self
    end

    def slice(*keys)
      keys.map! { |key| convert_key(key) }
      self.class.new(super)
    end

    def slice!(*keys)
      keys.map! { |key| convert_key(key) }
      super
    end

    def compact
      dup.tap(&:compact!)
    end

    # Convert to a regular hash with string keys.
    def to_hash
      copy = Hash[self]
      copy.transform_values! { |v| convert_value_to_hash(v) }
      set_defaults(copy)
      copy
    end

    def to_proc
      proc { |key| self[key] }
    end

    private
      def convert_key(key)
        Symbol === key ? key.name : key
      end

      def convert_value(value, conversion: nil)
        if value.is_a? Hash
          value.nested_under_indifferent_access
        elsif value.is_a?(Array)
          if conversion != :assignment || value.frozen?
            value = value.dup
          end
          value.map! { |e| convert_value(e, conversion: conversion) }
        else
          value
        end
      end

      def load_target
        if find_target?
          @target = merge_target_lists(find_target, target)
        elsif target.empty? && set_through_target_for_new_record?
          reflections = reflection.chain.reverse!

          @target = reflections.each_cons(2).reduce(through_association.target) do |middle_target, (middle_reflection, through_reflection)|
            if middle_target.nil? || (middle_reflection.collection? && middle_target.empty?)
              break []
            elsif middle_reflection.collection?
              middle_target.flat_map { |record| record.association(through_reflection.source_reflection_name).load_target }.compact
            else
              middle_target.association(through_reflection.source_reflection_name).load_target
            end

      def redirect(*args, &block)
        options = args.extract_options!
        status  = options.delete(:status) || 301
        path    = args.shift

        return OptionRedirect.new(status, options) if options.any?
        return PathRedirect.new(status, path) if String === path

        block = path if path.respond_to? :call
        raise ArgumentError, "redirection argument not supported" unless block
        Redirect.new status, block
      end

