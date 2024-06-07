Ev Kirası Tahmini 
Bu proje, ev kiralama fiyatlarını tahmin etmek için bir model geliştirme sürecini içerir. Veri seti, ev özelliklerini ve fiyatlarını içerir ve bu bilgileri kullanarak bir ev kiralama fiyatı tahminleme modeli oluşturulmuştur.

Veri Seti
Proje için kullanılan veri seti emlak.csv adlı bir CSV dosyasıdır. Veri seti şu özellikleri içerir:

Ev tipi (Daire, Müstakil, vb.)
Ev büyüklüğü
Kat sayısı
Oda sayısı
Isıtma tipi ve yakıtı
Yaş
Mobilyalı olup olmadığı
ve daha fazlası
Veri Temizleme ve Ön İşleme
Veri seti, eksik değerlerin, aykırı değerlerin, yanlış veri türlerinin ve diğer veri bütünlüğü sorunlarının düzeltilmesi için çeşitli ön işleme adımlarından geçirilmiştir. Bu adımlar arasında eksik değerlerin doldurulması, aykırı değerlerin ele alınması ve kategorik değişkenlerin kodlanması bulunmaktadır. Detaylar için kod ve yorumlara bakabilirsiniz.

Modelleme
Veri seti, bir CatBoost Regresyon modeli kullanılarak ev kiralama fiyatlarını tahmin etmek için eğitilmiştir. Model, çeşitli özellikleri ve ev fiyatlarını içeren veri seti üzerinde eğitilmiştir.

Model Değerlendirmesi
Model, 5 katlı çapraz doğrulama kullanılarak değerlendirilmiştir. Ayrıca, test veri seti üzerinde yapılan tahminlerle gerçek fiyatlar arasındaki farklar görselleştirilmiştir.

Dosyalar
emlak.csv: Kullanılan veri seti
model_pipeline.pkl: Eğitilmiş modelin seri hale getirilmiş versiyonu
Kullanım
Projeyi yerel bir ortamda çalıştırmak için aşağıdaki adımları izleyebilirsiniz:

Gerekli kütüphaneleri yükleyin: pip install -r requirements.txt
Proje dizinine gidin: cd project_directory
Kodları çalıştırın veya modeli kullanın: streamlit run app.py
