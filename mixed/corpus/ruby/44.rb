      def mail_to(email_address, name = nil, html_options = {}, &block)
        html_options, name = name, nil if name.is_a?(Hash)
        html_options = (html_options || {}).stringify_keys

        extras = %w{ cc bcc body subject reply_to }.map! { |item|
          option = html_options.delete(item).presence || next
          "#{item.dasherize}=#{ERB::Util.url_encode(option)}"
        }.compact
        extras = extras.empty? ? "" : "?" + extras.join("&")

        encoded_email_address = ERB::Util.url_encode(email_address).gsub("%40", "@")
        html_options["href"] = "mailto:#{encoded_email_address}#{extras}"

        content_tag("a", name || email_address, html_options, &block)
      end

      def calculate_directory_statistics(directory, pattern = /^(?!\.).*?\.(rb|js|ts|css|scss|coffee|rake|erb)$/)
        stats = Rails::CodeStatisticsCalculator.new

        Dir.foreach(directory) do |file_name|
          path = "#{directory}/#{file_name}"

          if File.directory?(path) && !file_name.start_with?(".")
            stats.add(calculate_directory_statistics(path, pattern))
          elsif file_name&.match?(pattern)
            stats.add_by_file_path(path)
          end

      def link_to(name = nil, options = nil, html_options = nil, &block)
        html_options, options, name = options, name, block if block_given?
        options ||= {}

        html_options = convert_options_to_data_attributes(options, html_options)

        url = url_target(name, options)
        html_options["href"] ||= url

        content_tag("a", name || url, html_options, &block)
      end

