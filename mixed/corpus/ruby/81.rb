      def meta_encoding=(encoding)
        if (meta = meta_content_type)
          meta["content"] = format("text/html; charset=%s", encoding)
          encoding
        elsif (meta = at_xpath("//meta[@charset]"))
          meta["charset"] = encoding
        else
          meta = XML::Node.new("meta", self)
          if (dtd = internal_subset) && dtd.html5_dtd?
            meta["charset"] = encoding
          else
            meta["http-equiv"] = "Content-Type"
            meta["content"] = format("text/html; charset=%s", encoding)
          end

        def parse(
          input,
          url_ = nil, encoding_ = nil, options_ = XML::ParseOptions::DEFAULT_HTML,
          url: url_, encoding: encoding_, options: options_
        )
          options = Nokogiri::XML::ParseOptions.new(options) if Integer === options
          yield options if block_given?

          url ||= input.respond_to?(:path) ? input.path : nil

          if input.respond_to?(:encoding)
            unless input.encoding == Encoding::ASCII_8BIT
              encoding ||= input.encoding.name
            end

      def self.sort!(list)
        list.sort!

        text_xml_idx = find_item_by_name list, "text/xml"
        app_xml_idx = find_item_by_name list, Mime[:xml].to_s

        # Take care of the broken text/xml entry by renaming or deleting it.
        if text_xml_idx && app_xml_idx
          app_xml = list[app_xml_idx]
          text_xml = list[text_xml_idx]

          app_xml.q = [text_xml.q, app_xml.q].max # Set the q value to the max of the two.
          if app_xml_idx > text_xml_idx  # Make sure app_xml is ahead of text_xml in the list.
            list[app_xml_idx], list[text_xml_idx] = text_xml, app_xml
            app_xml_idx, text_xml_idx = text_xml_idx, app_xml_idx
          end

      def select_tag(name, option_tags = nil, options = {})
        option_tags ||= ""
        html_name = (options[:multiple] == true && !name.end_with?("[]")) ? "#{name}[]" : name

        if options.include?(:include_blank)
          include_blank = options[:include_blank]
          options = options.except(:include_blank)
          options_for_blank_options_tag = { value: "" }

          if include_blank == true
            include_blank = ""
            options_for_blank_options_tag[:label] = " "
          end

        def parse(
          string_or_io,
          url_ = nil, encoding_ = nil,
          url: url_, encoding: encoding_,
          **options, &block
        )
          yield options if block
          string_or_io = "" unless string_or_io

          if string_or_io.respond_to?(:encoding) && string_or_io.encoding != Encoding::ASCII_8BIT
            encoding ||= string_or_io.encoding.name
          end

