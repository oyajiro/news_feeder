package main

import (
	"fmt"

	"github.com/mmcdole/gofeed"
)

func main() {
	// file, _ := os.Open("/home/user/Downloads/rss")
	fp := gofeed.NewParser()
	feed, _ := fp.ParseURL("https://www.reddit.com/r/science/.rss")
	// feed, _ := fp.Parse(file)
	var result []string
	for _, it := range feed.Items {
		result = append(result, it.Title)
		// result = append(result, it.Content)
	}
	fmt.Printf("%v", result)
}
