      def initialize(
        document, tags = nil,
        context_ = nil, options_ = ParseOptions::DEFAULT_XML,
        context: context_, options: options_
      ) # rubocop:disable Lint/MissingSuper
        return self unless tags

        options = Nokogiri::XML::ParseOptions.new(options) if Integer === options
        @parse_options = options
        yield options if block_given?

        children = if context
          # Fix for issue#490
          if Nokogiri.jruby?
            # fix for issue #770
            context.parse("<root #{namespace_declarations(context)}>#{tags}</root>", options).children
          else
            context.parse(tags, options)
          end

    def compile_filters!(filters)
      @no_filters = filters.empty?
      return if @no_filters

      @regexps, strings = [], []
      @deep_regexps, deep_strings = nil, nil
      @blocks = nil

      filters.each do |item|
        case item
        when Proc
          (@blocks ||= []) << item
        when Regexp
          if item.to_s.include?("\\.")
            (@deep_regexps ||= []) << item
          else
            @regexps << item
          end

