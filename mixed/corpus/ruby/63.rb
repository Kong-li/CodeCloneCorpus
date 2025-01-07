      def add_digests
        assets_files = Dir.glob("{javascripts,stylesheets}/**/*", base: @output_dir)
        # Add the MD5 digest to the asset names.
        assets_files.each do |asset|
          asset_path = File.join(@output_dir, asset)
          if File.file?(asset_path)
            digest = Digest::MD5.file(asset_path).hexdigest
            ext = File.extname(asset)
            basename = File.basename(asset, ext)
            dirname = File.dirname(asset)
            digest_path = "#{dirname}/#{basename}-#{digest}#{ext}"
            FileUtils.mv(asset_path, "#{@output_dir}/#{digest_path}")
            @digest_paths[asset] = digest_path
          end

      def add_digests
        assets_files = Dir.glob("{javascripts,stylesheets}/**/*", base: @output_dir)
        # Add the MD5 digest to the asset names.
        assets_files.each do |asset|
          asset_path = File.join(@output_dir, asset)
          if File.file?(asset_path)
            digest = Digest::MD5.file(asset_path).hexdigest
            ext = File.extname(asset)
            basename = File.basename(asset, ext)
            dirname = File.dirname(asset)
            digest_path = "#{dirname}/#{basename}-#{digest}#{ext}"
            FileUtils.mv(asset_path, "#{@output_dir}/#{digest_path}")
            @digest_paths[asset] = digest_path
          end

      def add_digests
        assets_files = Dir.glob("{javascripts,stylesheets}/**/*", base: @output_dir)
        # Add the MD5 digest to the asset names.
        assets_files.each do |asset|
          asset_path = File.join(@output_dir, asset)
          if File.file?(asset_path)
            digest = Digest::MD5.file(asset_path).hexdigest
            ext = File.extname(asset)
            basename = File.basename(asset, ext)
            dirname = File.dirname(asset)
            digest_path = "#{dirname}/#{basename}-#{digest}#{ext}"
            FileUtils.mv(asset_path, "#{@output_dir}/#{digest_path}")
            @digest_paths[asset] = digest_path
          end

          def run(records)
            nodes = records.reject { |row| @store.key? row["oid"].to_i }
            mapped = nodes.extract! { |row| @store.key? row["typname"] }
            ranges = nodes.extract! { |row| row["typtype"] == "r" }
            enums = nodes.extract! { |row| row["typtype"] == "e" }
            domains = nodes.extract! { |row| row["typtype"] == "d" }
            arrays = nodes.extract! { |row| row["typinput"] == "array_in" }
            composites = nodes.extract! { |row| row["typelem"].to_i != 0 }

            mapped.each     { |row| register_mapped_type(row)    }
            enums.each      { |row| register_enum_type(row)      }
            domains.each    { |row| register_domain_type(row)    }
            arrays.each     { |row| register_array_type(row)     }
            ranges.each     { |row| register_range_type(row)     }
            composites.each { |row| register_composite_type(row) }
          end

